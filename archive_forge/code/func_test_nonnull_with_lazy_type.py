from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_nonnull_with_lazy_type():
    MyType = object()
    field = NonNull(lambda: MyType)
    assert field.of_type == MyType