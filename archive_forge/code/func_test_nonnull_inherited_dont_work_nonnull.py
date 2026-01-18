from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_nonnull_inherited_dont_work_nonnull():
    with raises(Exception) as exc_info:
        NonNull(NonNull(String))
    assert str(exc_info.value) == 'Can only create NonNull of a Nullable GraphQLType but got: String!.'