import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class LifoMemoryPriorityQueueTest(PQueueTestMixin, LifoTestMixin, QueuelibTestCase):

    def qfactory(self, prio):
        return track_closed(LifoMemoryQueue)()