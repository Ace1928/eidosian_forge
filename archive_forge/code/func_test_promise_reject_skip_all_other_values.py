from pytest import raises
from promise import Promise
from promise.promise_list import PromiseList
def test_promise_reject_skip_all_other_values():
    e1 = Exception('Error1')
    e2 = Exception('Error2')
    p = Promise()
    all_promises = all([1, Promise.reject(e1), Promise.reject(e2)])
    with raises(Exception) as exc_info:
        all_promises.get()
    assert str(exc_info.value) == 'Error1'