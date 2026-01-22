from collections import deque
import select
import msgpack
def _enqueue_incoming_notification(self, m):
    self._notifications.append(m)
    self._incoming += 1