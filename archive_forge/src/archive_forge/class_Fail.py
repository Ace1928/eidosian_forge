import pytest
from IPython.core.error import TryNext
from IPython.core.hooks import CommandChainDispatcher
class Fail(object):

    def __init__(self, message):
        self.message = message
        self.called = False

    def __call__(self):
        self.called = True
        raise TryNext(self.message)