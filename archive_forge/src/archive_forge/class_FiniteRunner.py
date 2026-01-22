import abc
from automaton import exceptions as excp
from automaton import machines
class FiniteRunner(Runner):
    """Finite machine runner used to run a finite machine.

    Only **one** runner per machine should be active at the same time (aka
    there should not be multiple runners using the same machine instance at
    the same time).
    """

    def __init__(self, machine):
        """Create a runner for the given machine."""
        if not isinstance(machine, (machines.FiniteMachine,)):
            raise TypeError('FiniteRunner only works with FiniteMachine(s)')
        super(FiniteRunner, self).__init__(machine)

    def run(self, event, initialize=True):
        for transition in self.run_iter(event, initialize=initialize):
            pass

    def run_iter(self, event, initialize=True):
        if initialize:
            self._machine.initialize()
        while True:
            old_state = self._machine.current_state
            reaction, terminal = self._machine.process_event(event)
            new_state = self._machine.current_state
            try:
                sent_event = (yield (old_state, new_state))
            except GeneratorExit:
                break
            if terminal:
                break
            if reaction is None and sent_event is None:
                raise excp.NotFound(_JUMPER_NOT_FOUND_TPL % (new_state, old_state, event))
            elif sent_event is not None:
                event = sent_event
            else:
                cb, args, kwargs = reaction
                event = cb(old_state, new_state, event, *args, **kwargs)