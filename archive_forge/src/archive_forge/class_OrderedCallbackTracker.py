from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
class OrderedCallbackTracker:
    """
    Test helper for tracking callbacks and their order.
    """

    def __init__(self):
        self.callList = []

    def call1(self, object):
        self.callList.append(self.call1)

    def call2(self, object):
        self.callList.append(self.call2)

    def call3(self, object):
        self.callList.append(self.call3)