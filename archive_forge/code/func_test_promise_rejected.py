from pytest import raises
from promise import Promise
from promise.promise_list import PromiseList
def test_promise_rejected():
    e = Exception('Error')
    all_promises = all([1, 2, Promise.reject(e)])
    with raises(Exception) as exc_info:
        all_promises.get()
    assert str(exc_info.value) == 'Error'