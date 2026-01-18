from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_objecttype_with_possible_types_and_is_type_of_should_raise():
    with raises(AssertionError) as excinfo:

        class MyObjectType(ObjectType):

            class Meta:
                possible_types = (dict,)

            @classmethod
            def is_type_of(cls, root, context, info):
                return False
    assert str(excinfo.value) == 'MyObjectType.Meta.possible_types will cause type collision with MyObjectType.is_type_of. Please use one or other.'