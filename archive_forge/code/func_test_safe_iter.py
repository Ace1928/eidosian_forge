from tune._utils import _EMPTY_ITER, dict_product, product, safe_iter
def test_safe_iter():
    assert [1] == list(safe_iter([1]))
    assert [1] == list(safe_iter(safe_iter([1])))
    assert [None] == list(safe_iter(safe_iter([None])))
    assert [1] == list(safe_iter([1], safe=False))
    assert [_EMPTY_ITER] == list(safe_iter([]))
    assert [_EMPTY_ITER] == list(safe_iter(safe_iter([])))
    assert [] == list(safe_iter([], safe=False))