from promise import Promise
from .utils import assert_exception
from threading import Event
def test_promise_resolved_after():
    """
    The first argument to 'then' must be called when a promise is
    fulfilled.
    """
    c = Counter()

    def check(v, c):
        assert v == 5
        c.tick()
    p1 = Promise()
    p2 = p1.then(lambda v: check(v, c))
    p1.do_resolve(5)
    Promise.wait(p2)
    assert 1 == c.value()