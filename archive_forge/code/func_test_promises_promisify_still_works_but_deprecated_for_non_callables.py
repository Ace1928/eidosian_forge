from time import sleep
from pytest import raises, fixture
from threading import Event
from promise import (
from concurrent.futures import Future
from threading import Thread
from .utils import assert_exception
def test_promises_promisify_still_works_but_deprecated_for_non_callables():
    x = promisify(1)
    assert isinstance(x, Promise)
    assert x.get() == 1