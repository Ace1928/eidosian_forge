from pytest import raises
from promise import Promise
from promise.promise_list import PromiseList
def test_promise_mixed():
    all_promises = all([1, 2, Promise.resolve(3)])
    assert all_promises.get() == [1, 2, 3]