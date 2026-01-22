import itertools
import time
from oslo_utils import timeutils
from taskflow.engines.action_engine import compiler as co
from taskflow import exceptions as exc
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
class DurationListener(base.Listener):
    """Listener that captures task duration.

    It records how long a task took to execute (or fail)
    to storage. It saves the duration in seconds as float value
    to task metadata with key ``'duration'``.
    """

    def __init__(self, engine):
        super(DurationListener, self).__init__(engine, task_listen_for=WATCH_STATES, flow_listen_for=WATCH_STATES)
        self._timers = {co.TASK: {}, co.FLOW: {}}

    def deregister(self):
        super(DurationListener, self).deregister()
        for item_type, timers in self._timers.items():
            leftover_timers = len(timers)
            if leftover_timers:
                LOG.warning('%s %s(s) did not enter %s states', leftover_timers, item_type, FINISHED_STATES)
            timers.clear()

    def _record_ending(self, timer, item_type, item_name, state):
        meta_update = {'duration': timer.elapsed()}
        try:
            storage = self._engine.storage
            if item_type == co.FLOW:
                storage.update_flow_metadata(meta_update)
            else:
                storage.update_atom_metadata(item_name, meta_update)
        except exc.StorageFailure:
            LOG.warning('Failure to store duration update %s for %s %s', meta_update, item_type, item_name, exc_info=True)

    def _task_receiver(self, state, details):
        task_name = details['task_name']
        self._receiver(co.TASK, task_name, state)

    def _flow_receiver(self, state, details):
        flow_name = details['flow_name']
        self._receiver(co.FLOW, flow_name, state)

    def _receiver(self, item_type, item_name, state):
        if state == states.PENDING:
            self._timers[item_type].pop(item_name, None)
        elif state in STARTING_STATES:
            self._timers[item_type][item_name] = timeutils.StopWatch().start()
        elif state in FINISHED_STATES:
            timer = self._timers[item_type].pop(item_name, None)
            if timer is not None:
                timer.stop()
                self._record_ending(timer, item_type, item_name, state)