from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_rejected():
    p = Promise.rejected(Exception('Static rejected'))
    assert p.is_rejected
    with raises(Exception) as exc_info:
        p.get()
    assert str(exc_info.value) == 'Static rejected'