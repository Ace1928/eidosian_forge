from textwrap import dedent
from pytest import raises
from graphql.type import GraphQLObjectType, GraphQLSchema
from ..field import Field
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
def test_schema_requires_query_type():
    schema = Schema()
    result = schema.execute('query {}')
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.message == 'Query root type must be provided.'