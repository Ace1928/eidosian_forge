import unittest
import inspect
import threading
def assertSent(self, entity):
    self.send(entity)
    try:
        self.assertEqual(entity.toProtocolTreeNode(), self.lowerSink.pop())
    except IndexError:
        raise AssertionError("Entity '%s' was not sent through this layer" % entity.getTag())