import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
@pytest.fixture
def typedef_schema():
    return Client(type_def='\nschema {\n  query: Query\n}\n\ninterface Character {\n  appearsIn: [Episode]\n  friends: [Character]\n  id: String!\n  name: String\n}\n\ntype Droid implements Character {\n  appearsIn: [Episode]\n  friends: [Character]\n  id: String!\n  name: String\n  primaryFunction: String\n}\n\nenum Episode {\n  EMPIRE\n  JEDI\n  NEWHOPE\n}\n\ntype Human implements Character {\n  appearsIn: [Episode]\n  friends: [Character]\n  homePlanet: String\n  id: String!\n  name: String\n}\n\ntype Query {\n  droid(id: String!): Droid\n  hero(episode: Episode): Character\n  human(id: String!): Human\n}')