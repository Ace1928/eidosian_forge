import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def test_require_fields_on_object(client):
    query = '\n        query HeroNoFieldsQuery {\n          hero\n        }\n    '
    assert validation_errors(client, query)