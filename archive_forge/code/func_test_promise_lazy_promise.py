from pytest import raises
from promise import Promise
from promise.promise_list import PromiseList
def test_promise_lazy_promise():
    p = Promise()
    all_promises = all([1, 2, p])
    assert not all_promises.is_fulfilled
    p.do_resolve(3)
    assert all_promises.get() == [1, 2, 3]