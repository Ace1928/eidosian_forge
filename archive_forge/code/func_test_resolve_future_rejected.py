from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_resolve_future_rejected(resolve):
    future = Future()
    promise = resolve(future)
    assert promise.is_pending
    future.set_exception(Exception('Future rejected'))
    assert promise.is_rejected
    assert_exception(promise.reason, Exception, 'Future rejected')