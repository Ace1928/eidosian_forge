from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
class CallbackTracker:
    """
    Test helper for tracking callbacks.

    Increases a counter on each call to L{call} and stores the object
    passed in the call.
    """

    def __init__(self):
        self.called = 0
        self.obj = None

    def call(self, obj):
        self.called = self.called + 1
        self.obj = obj