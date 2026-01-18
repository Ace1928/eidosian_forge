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
def test_objecttype():

    class MyObjectType(ObjectType):
        """Description"""
        foo = String(bar=String(description='Argument description', default_value='x'), description='Field description')
        bar = String(name='gizmo')

        def resolve_foo(self, bar):
            return bar
    type_map = create_type_map([MyObjectType])
    assert 'MyObjectType' in type_map
    graphql_type = type_map['MyObjectType']
    assert isinstance(graphql_type, GraphQLObjectType)
    assert graphql_type.name == 'MyObjectType'
    assert graphql_type.description == 'Description'
    fields = graphql_type.fields
    assert list(fields) == ['foo', 'gizmo']
    foo_field = fields['foo']
    assert isinstance(foo_field, GraphQLField)
    assert foo_field.description == 'Field description'
    assert foo_field.args == {'bar': GraphQLArgument(GraphQLString, description='Argument description', default_value='x', out_name='bar')}