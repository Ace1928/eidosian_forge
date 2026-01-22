from zaqarclient.queues.v1 import claim
from zaqarclient.queues.v1 import iterator as iterate
from zaqarclient.queues.v2 import core
from zaqarclient.queues.v2 import message
class Claim(claim.Claim):

    def _create(self):
        req, trans = self._queue.client._request_and_transport()
        msgs = core.claim_create(trans, req, self._queue._name, ttl=self._ttl, grace=self._grace, limit=self._limit)
        if msgs is not None:
            if self._queue.client.api_version >= 1.1:
                msgs = msgs['messages']
            self.id = msgs[0]['href'].split('=')[-1]
        self._message_iter = iterate._Iterator(self._queue.client, msgs or [], 'messages', message.create_object(self._queue))