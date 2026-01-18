from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_as_container_args_kwargs():
    container = Container('1', field2='2')
    assert container.field1 == '1'
    assert container.field2 == '2'