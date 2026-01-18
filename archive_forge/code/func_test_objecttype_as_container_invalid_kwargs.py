from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_as_container_invalid_kwargs():
    msg = "__init__\\(\\) got an unexpected keyword argument 'unexisting_field'"
    with raises(TypeError, match=msg):
        Container(unexisting_field='3')