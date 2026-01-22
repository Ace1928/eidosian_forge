import itertools
import time
from oslo_utils import timeutils
from taskflow.engines.action_engine import compiler as co
from taskflow import exceptions as exc
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
class EventTimeListener(base.Listener):
    """Listener that captures task, flow, and retry event timestamps.

    It records how when an event is received (using unix time) to
    storage. It saves the timestamps under keys (in atom or flow details
    metadata) of the format ``{event}-timestamp`` where ``event`` is the
    state/event name that has been received.

    This information can be later extracted/examined to derive durations...
    """

    def __init__(self, engine, task_listen_for=base.DEFAULT_LISTEN_FOR, flow_listen_for=base.DEFAULT_LISTEN_FOR, retry_listen_for=base.DEFAULT_LISTEN_FOR):
        super(EventTimeListener, self).__init__(engine, task_listen_for=task_listen_for, flow_listen_for=flow_listen_for, retry_listen_for=retry_listen_for)

    def _record_atom_event(self, state, atom_name):
        meta_update = {'%s-timestamp' % state: time.time()}
        try:
            self._engine.storage.update_atom_metadata(atom_name, meta_update)
        except exc.StorageFailure:
            LOG.warning('Failure to store timestamp %s for atom %s', meta_update, atom_name, exc_info=True)

    def _flow_receiver(self, state, details):
        meta_update = {'%s-timestamp' % state: time.time()}
        try:
            self._engine.storage.update_flow_metadata(meta_update)
        except exc.StorageFailure:
            LOG.warning('Failure to store timestamp %s for flow %s', meta_update, details['flow_name'], exc_info=True)

    def _task_receiver(self, state, details):
        self._record_atom_event(state, details['task_name'])

    def _retry_receiver(self, state, details):
        self._record_atom_event(state, details['retry_name'])