from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_observerRaisingException(self):
    """
        Test that exceptions in observers do not bubble up to dispatch.

        The exceptions raised in observers should be logged and other
        observers should be called as if nothing happened.
        """

    class OrderedCallbackList(utility.CallbackList):

        def __init__(self):
            self.callbacks = OrderedDict()

    class TestError(Exception):
        pass

    def raiseError(_):
        raise TestError()
    d = EventDispatcher()
    cb = CallbackTracker()
    originalCallbackList = utility.CallbackList
    try:
        utility.CallbackList = OrderedCallbackList
        d.addObserver('//event/test', raiseError)
        d.addObserver('//event/test', cb.call)
        try:
            d.dispatch(None, '//event/test')
        except TestError:
            self.fail('TestError raised. Should have been logged instead.')
        self.assertEqual(1, len(self.flushLoggedErrors(TestError)))
        self.assertEqual(1, cb.called)
    finally:
        utility.CallbackList = originalCallbackList