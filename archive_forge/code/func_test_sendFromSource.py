from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_sendFromSource(self):
    """
        Send an element from the source and observe it from the sink.
        """

    def cb(obj):
        called.append(obj)
    called = []
    self.pipe.sink.addObserver('/test[@xmlns="testns"]', cb)
    element = Element(('testns', 'test'))
    self.pipe.source.send(element)
    self.assertEqual([element], called)