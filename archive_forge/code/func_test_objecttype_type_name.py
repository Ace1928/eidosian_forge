from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_type_name():

    class MyObjectType(ObjectType, name='FooType'):
        pass
    assert MyObjectType._meta.name == 'FooType'