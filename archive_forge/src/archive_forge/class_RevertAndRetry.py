import abc
import weakref
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import executor as ex
from taskflow import logging
from taskflow import retry as retry_atom
from taskflow import states as st
class RevertAndRetry(Strategy):
    """Sets the *associated* subflow for revert to be later retried."""
    strategy = retry_atom.RETRY

    def __init__(self, runtime, retry):
        super(RevertAndRetry, self).__init__(runtime)
        self._retry = retry

    def apply(self):
        tweaked = self._runtime.reset_atoms([self._retry], state=None, intention=st.RETRY)
        tweaked.extend(self._runtime.reset_subgraph(self._retry, state=None, intention=st.REVERT))
        return tweaked