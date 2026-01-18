from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promise_all_if():
    p1 = Promise()
    p2 = Promise()
    pd1 = Promise.all([p1, p2])
    pd2 = Promise.all([p1])
    pd3 = Promise.all([])
    pd3._wait()
    assert p1.is_pending
    assert p2.is_pending
    assert pd1.is_pending
    assert pd2.is_pending
    assert pd3.is_fulfilled
    p1.do_resolve(5)
    p1._wait()
    pd2._wait()
    assert p1.is_fulfilled
    assert p2.is_pending
    assert pd1.is_pending
    assert pd2.is_fulfilled
    p2.do_resolve(10)
    p2._wait()
    pd1._wait()
    pd2._wait()
    assert p1.is_fulfilled
    assert p2.is_fulfilled
    assert pd1.is_fulfilled
    assert pd2.is_fulfilled
    assert 5 == p1.get()
    assert 10 == p2.get()
    assert 5 == pd1.get()[0]
    assert 5 == pd2.get()[0]
    assert 10 == pd1.get()[1]
    assert [] == pd3.get()