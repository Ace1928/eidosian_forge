from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_nonnull_inherited_works_list():
    _list = NonNull(List(String))
    assert isinstance(_list.of_type, List)
    assert _list.of_type.of_type == String