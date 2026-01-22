from collections import deque
import select
import msgpack
def _enqueue_incoming_response(self, m):
    msgid, error, result = m
    try:
        self._pending_requests.remove(msgid)
    except KeyError:
        return
    assert msgid not in self._responses
    self._responses[msgid] = (error, result)
    self._incoming += 1