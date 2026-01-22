from concurrent import futures
import weakref
from automaton import machines
from oslo_utils import timeutils
from taskflow import logging
from taskflow import states as st
from taskflow.types import failure
from taskflow.utils import iter_utils
class MachineBuilder(object):
    """State machine *builder* that powers the engine components.

    NOTE(harlowja): the machine (states and events that will trigger
    transitions) that this builds is represented by the following
    table::

        +--------------+------------------+------------+----------+---------+
        |    Start     |      Event       |    End     | On Enter | On Exit |
        +--------------+------------------+------------+----------+---------+
        |  ANALYZING   |    completed     | GAME_OVER  |    .     |    .    |
        |  ANALYZING   |  schedule_next   | SCHEDULING |    .     |    .    |
        |  ANALYZING   |  wait_finished   |  WAITING   |    .     |    .    |
        |  FAILURE[$]  |        .         |     .      |    .     |    .    |
        |  GAME_OVER   |      failed      |  FAILURE   |    .     |    .    |
        |  GAME_OVER   |     reverted     |  REVERTED  |    .     |    .    |
        |  GAME_OVER   |     success      |  SUCCESS   |    .     |    .    |
        |  GAME_OVER   |    suspended     | SUSPENDED  |    .     |    .    |
        |   RESUMING   |  schedule_next   | SCHEDULING |    .     |    .    |
        | REVERTED[$]  |        .         |     .      |    .     |    .    |
        |  SCHEDULING  |  wait_finished   |  WAITING   |    .     |    .    |
        |  SUCCESS[$]  |        .         |     .      |    .     |    .    |
        | SUSPENDED[$] |        .         |     .      |    .     |    .    |
        | UNDEFINED[^] |      start       |  RESUMING  |    .     |    .    |
        |   WAITING    | examine_finished | ANALYZING  |    .     |    .    |
        +--------------+------------------+------------+----------+---------+

    Between any of these yielded states (minus ``GAME_OVER`` and ``UNDEFINED``)
    if the engine has been suspended or the engine has failed (due to a
    non-resolveable task failure or scheduling failure) the machine will stop
    executing new tasks (currently running tasks will be allowed to complete)
    and this machines run loop will be broken.

    NOTE(harlowja): If the runtimes scheduler component is able to schedule
    tasks in parallel, this enables parallel running and/or reversion.
    """

    def __init__(self, runtime, waiter):
        self._runtime = weakref.proxy(runtime)
        self._selector = runtime.selector
        self._completer = runtime.completer
        self._scheduler = runtime.scheduler
        self._storage = runtime.storage
        self._waiter = waiter

    def build(self, statistics, timeout=None, gather_statistics=True):
        """Builds a state-machine (that is used during running)."""
        if gather_statistics:
            watches = {}
            state_statistics = {}
            statistics['seconds_per_state'] = state_statistics
            watches = {}
            for timed_state in TIMED_STATES:
                state_statistics[timed_state.lower()] = 0.0
                watches[timed_state] = timeutils.StopWatch()
            statistics['discarded_failures'] = 0
            statistics['awaiting'] = 0
            statistics['completed'] = 0
            statistics['incomplete'] = 0
        memory = MachineMemory()
        if timeout is None:
            timeout = WAITING_TIMEOUT
        do_complete = self._completer.complete
        do_complete_failure = self._completer.complete_failure
        get_atom_intention = self._storage.get_atom_intention

        def do_schedule(next_nodes):
            with self._storage.lock.write_lock():
                return self._scheduler.schedule(sorted(next_nodes, key=lambda node: getattr(node, 'priority', 0), reverse=True))

        def iter_next_atoms(atom=None, apply_deciders=True):
            maybe_atoms_it = self._selector.iter_next_atoms(atom=atom)
            for atom, late_decider in maybe_atoms_it:
                if apply_deciders:
                    proceed = late_decider.check_and_affect(self._runtime)
                    if proceed:
                        yield atom
                else:
                    yield atom

        def resume(old_state, new_state, event):
            with self._storage.lock.write_lock():
                memory.next_up.update(iter_utils.unique_seen((self._completer.resume(), iter_next_atoms())))
            return SCHEDULE

        def game_over(old_state, new_state, event):
            if memory.failures:
                return FAILED
            with self._storage.lock.read_lock():
                leftover_atoms = iter_utils.count(iter_next_atoms(apply_deciders=False))
            if leftover_atoms:
                LOG.trace('Suspension determined to have been reacted to since (at least) %s atoms have been left in an unfinished state', leftover_atoms)
                return SUSPENDED
            elif self._runtime.is_success():
                return SUCCESS
            else:
                return REVERTED

        def schedule(old_state, new_state, event):
            with self._storage.lock.write_lock():
                current_flow_state = self._storage.get_flow_state()
                if current_flow_state == st.RUNNING and memory.next_up:
                    not_done, failures = do_schedule(memory.next_up)
                    if not_done:
                        memory.not_done.update(not_done)
                    if failures:
                        memory.failures.extend(failures)
                    memory.next_up.intersection_update(not_done)
                elif current_flow_state == st.SUSPENDING and memory.not_done:
                    memory.cancel_futures()
            return WAIT

        def complete_an_atom(fut):
            atom = fut.atom
            try:
                outcome, result = fut.result()
                do_complete(atom, outcome, result)
                if isinstance(result, failure.Failure):
                    retain = do_complete_failure(atom, outcome, result)
                    if retain:
                        memory.failures.append(result)
                    else:
                        if LOG.isEnabledFor(logging.DEBUG):
                            intention = get_atom_intention(atom.name)
                            LOG.debug("Discarding failure '%s' (in response to outcome '%s') under completion units request during completion of atom '%s' (intention is to %s)", result, outcome, atom, intention)
                        if gather_statistics:
                            statistics['discarded_failures'] += 1
                if gather_statistics:
                    statistics['completed'] += 1
            except futures.CancelledError:
                return WAS_CANCELLED
            except Exception:
                memory.failures.append(failure.Failure())
                LOG.exception("Engine '%s' atom post-completion failed", atom)
                return FAILED_COMPLETING
            else:
                return SUCCESSFULLY_COMPLETED

        def wait(old_state, new_state, event):
            if memory.not_done:
                done, not_done = self._waiter(memory.not_done, timeout=timeout)
                memory.done.update(done)
                memory.not_done = not_done
            return ANALYZE

        def analyze(old_state, new_state, event):
            next_up = set()
            with self._storage.lock.write_lock():
                while memory.done:
                    fut = memory.done.pop()
                    completion_status = complete_an_atom(fut)
                    if not memory.failures and completion_status != WAS_CANCELLED:
                        atom = fut.atom
                        try:
                            more_work = set(iter_next_atoms(atom=atom))
                        except Exception:
                            memory.failures.append(failure.Failure())
                            LOG.exception("Engine '%s' atom post-completion next atom searching failed", atom)
                        else:
                            next_up.update(more_work)
            current_flow_state = self._storage.get_flow_state()
            if current_flow_state == st.RUNNING and next_up and (not memory.failures):
                memory.next_up.update(next_up)
                return SCHEDULE
            elif memory.not_done:
                if current_flow_state == st.SUSPENDING:
                    memory.cancel_futures()
                return WAIT
            else:
                return FINISH

        def on_exit(old_state, event):
            LOG.trace("Exiting old state '%s' in response to event '%s'", old_state, event)
            if gather_statistics:
                if old_state in watches:
                    w = watches[old_state]
                    w.stop()
                    state_statistics[old_state.lower()] += w.elapsed()
                if old_state in (st.SCHEDULING, st.WAITING):
                    statistics['incomplete'] = len(memory.not_done)
                if old_state in (st.ANALYZING, st.SCHEDULING):
                    statistics['awaiting'] = len(memory.next_up)

        def on_enter(new_state, event):
            LOG.trace("Entering new state '%s' in response to event '%s'", new_state, event)
            if gather_statistics and new_state in watches:
                watches[new_state].restart()
        state_kwargs = {'on_exit': on_exit, 'on_enter': on_enter}
        m = machines.FiniteMachine()
        m.add_state(GAME_OVER, **state_kwargs)
        m.add_state(UNDEFINED, **state_kwargs)
        m.add_state(st.ANALYZING, **state_kwargs)
        m.add_state(st.RESUMING, **state_kwargs)
        m.add_state(st.REVERTED, terminal=True, **state_kwargs)
        m.add_state(st.SCHEDULING, **state_kwargs)
        m.add_state(st.SUCCESS, terminal=True, **state_kwargs)
        m.add_state(st.SUSPENDED, terminal=True, **state_kwargs)
        m.add_state(st.WAITING, **state_kwargs)
        m.add_state(st.FAILURE, terminal=True, **state_kwargs)
        m.default_start_state = UNDEFINED
        m.add_transition(GAME_OVER, st.REVERTED, REVERTED)
        m.add_transition(GAME_OVER, st.SUCCESS, SUCCESS)
        m.add_transition(GAME_OVER, st.SUSPENDED, SUSPENDED)
        m.add_transition(GAME_OVER, st.FAILURE, FAILED)
        m.add_transition(UNDEFINED, st.RESUMING, START)
        m.add_transition(st.ANALYZING, GAME_OVER, FINISH)
        m.add_transition(st.ANALYZING, st.SCHEDULING, SCHEDULE)
        m.add_transition(st.ANALYZING, st.WAITING, WAIT)
        m.add_transition(st.RESUMING, st.SCHEDULING, SCHEDULE)
        m.add_transition(st.SCHEDULING, st.WAITING, WAIT)
        m.add_transition(st.WAITING, st.ANALYZING, ANALYZE)
        m.add_reaction(GAME_OVER, FINISH, game_over)
        m.add_reaction(st.ANALYZING, ANALYZE, analyze)
        m.add_reaction(st.RESUMING, START, resume)
        m.add_reaction(st.SCHEDULING, SCHEDULE, schedule)
        m.add_reaction(st.WAITING, WAIT, wait)
        m.freeze()
        return (m, memory)