from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_resolve_promise_subclass():

    class MyPromise(Promise):
        pass
    p = Promise()
    p.do_resolve(10)
    m_p = MyPromise.resolve(p)
    assert isinstance(m_p, MyPromise)
    assert m_p.get() == p.get()