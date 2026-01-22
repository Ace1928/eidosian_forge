from collections import deque
import select
import msgpack
def _enqueue_incoming_request(self, m):
    self._requests.append(m)
    self._incoming += 1