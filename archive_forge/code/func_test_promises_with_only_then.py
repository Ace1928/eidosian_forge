from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promises_with_only_then():
    context = {'success': False}
    error = RuntimeError('Ooops!')
    promise1 = Promise(lambda resolve, reject: context.update({'promise1_reject': reject}))
    promise2 = promise1.then(lambda x: None)
    promise3 = promise1.then(lambda x: None)
    context['promise1_reject'](error)
    promise2._wait()
    promise3._wait()
    assert promise2.reason == error
    assert promise3.reason == error