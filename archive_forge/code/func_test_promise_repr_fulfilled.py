from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promise_repr_fulfilled():
    val = {1: 2}
    promise = Promise.fulfilled(val)
    promise._wait()
    assert repr(promise) == '<Promise at {} fulfilled with {}>'.format(hex(id(promise)), repr(val))