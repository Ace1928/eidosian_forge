from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_nonnull_with_unmounted_type():
    with raises(Exception) as exc_info:
        NonNull(String())
    assert str(exc_info.value) == 'NonNull could not have a mounted String() as inner type. Try with NonNull(String).'