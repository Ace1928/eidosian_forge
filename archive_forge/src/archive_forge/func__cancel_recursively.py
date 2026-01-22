import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
def _cancel_recursively(self, key, runner):
    try:
        runner.cancel()
    except Exception as ex:
        LOG.debug('Exception cancelling task: %s', str(ex))
    node = self._graph[key]
    for dependent_node in node.required_by():
        node_runner = self._runners[dependent_node]
        self._cancel_recursively(dependent_node, node_runner)
    del self._graph[key]