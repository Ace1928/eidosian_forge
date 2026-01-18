from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_resolve_future_like(resolve):

    class CustomThenable(object):

        def add_done_callback(self, f):
            f(True)

        def done(self):
            return True

        def exception(self):
            pass

        def result(self):
            return True
    instance = CustomThenable()
    promise = resolve(instance)
    assert promise.get() == True