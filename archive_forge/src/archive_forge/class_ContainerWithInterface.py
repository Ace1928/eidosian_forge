from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
class ContainerWithInterface(ObjectType):

    class Meta:
        interfaces = (MyInterface,)
    field1 = Field(MyType)
    field2 = Field(MyType)