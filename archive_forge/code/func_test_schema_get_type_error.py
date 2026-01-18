from textwrap import dedent
from pytest import raises
from graphql.type import GraphQLObjectType, GraphQLSchema
from ..field import Field
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
def test_schema_get_type_error():
    schema = Schema(Query)
    with raises(AttributeError) as exc_info:
        schema.X
    assert str(exc_info.value) == 'Type "X" not found in the Schema'