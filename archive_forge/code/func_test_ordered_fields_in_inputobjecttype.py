from graphql import Undefined
from ..argument import Argument
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..objecttype import ObjectType
from ..scalars import Boolean, String
from ..schema import Schema
from ..unmountedtype import UnmountedType
from ... import NonNull
def test_ordered_fields_in_inputobjecttype():

    class MyInputObjectType(InputObjectType):
        b = InputField(MyType)
        a = InputField(MyType)
        field = MyScalar()
        asa = InputField(MyType)
    assert list(MyInputObjectType._meta.fields) == ['b', 'a', 'field', 'asa']