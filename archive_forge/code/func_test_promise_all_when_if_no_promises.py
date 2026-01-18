from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promise_all_when_if_no_promises():
    pl = Promise.all([10, 32, False, True])
    assert pl.get() == [10, 32, False, True]