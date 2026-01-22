import itertools
import logging
from oslo_messaging.notify import dispatcher as notify_dispatcher
from oslo_messaging import server as msg_server
from oslo_messaging import transport as msg_transport
class NotificationServerBase(msg_server.MessageHandlingServer):

    def __init__(self, transport, targets, dispatcher, executor=None, allow_requeue=True, pool=None, batch_size=1, batch_timeout=None):
        super(NotificationServerBase, self).__init__(transport, dispatcher, executor)
        self._allow_requeue = allow_requeue
        self._pool = pool
        self.targets = targets
        self._targets_priorities = set(itertools.product(self.targets, self.dispatcher.supported_priorities))
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout

    def _create_listener(self):
        return self.transport._listen_for_notifications(self._targets_priorities, self._pool, self._batch_size, self._batch_timeout)