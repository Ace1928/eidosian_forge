from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_resolve_future(resolve):
    future = Future()
    promise = resolve(future)
    assert promise.is_pending
    future.set_result(1)
    assert promise.get() == 1
    assert promise.is_fulfilled