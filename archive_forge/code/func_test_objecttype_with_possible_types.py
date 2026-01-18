from graphql import Undefined
from graphql.type import (
from ..dynamic import Dynamic
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Int, String
from ..schema import Schema
from ..structures import List, NonNull
def test_objecttype_with_possible_types():

    class MyObjectType(ObjectType):
        """Description"""

        class Meta:
            possible_types = (dict,)
        foo_bar = String()
    type_map = create_type_map([MyObjectType])
    graphql_type = type_map['MyObjectType']
    assert graphql_type.is_type_of
    assert graphql_type.is_type_of({}, None) is True
    assert graphql_type.is_type_of(MyObjectType(), None) is False