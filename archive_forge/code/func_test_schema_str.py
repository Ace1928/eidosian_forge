from textwrap import dedent
from pytest import raises
from graphql.type import GraphQLObjectType, GraphQLSchema
from ..field import Field
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
def test_schema_str():
    schema = Schema(Query)
    assert str(schema).strip() == dedent('\n        type Query {\n          inner: MyOtherType\n        }\n\n        type MyOtherType {\n          field: String\n        }\n        ').strip()