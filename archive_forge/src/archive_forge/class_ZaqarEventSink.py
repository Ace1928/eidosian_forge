from oslo_log import log as logging
from zaqarclient.queues.v2 import client as zaqarclient
from zaqarclient.transport import errors as zaqar_errors
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
class ZaqarEventSink(object):

    def __init__(self, target, ttl=None):
        self._target = target
        self._ttl = ttl

    def consume(self, context, event):
        zaqar_plugin = context.clients.client_plugin('zaqar')
        zaqar = zaqar_plugin.client()
        queue = zaqar.queue(self._target, auto_create=False)
        ttl = self._ttl if self._ttl is not None else zaqar_plugin.DEFAULT_TTL
        queue.post({'body': event, 'ttl': ttl})