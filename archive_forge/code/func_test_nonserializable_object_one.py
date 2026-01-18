import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_nonserializable_object_one(self):
    self.assertRaises(TypeError, self.q.push, lambda x: x, 0)
    self.assertEqual(self.q.close(), [])