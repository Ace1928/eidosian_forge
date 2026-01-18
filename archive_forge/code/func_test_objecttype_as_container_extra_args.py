from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_as_container_extra_args():
    msg = '__init__\\(\\) takes from 1 to 3 positional arguments but 4 were given'
    with raises(TypeError, match=msg):
        Container('1', '2', '3')