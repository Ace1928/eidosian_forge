from pytest import mark
from graphql_relay.utils import base64
from graphene.types import ObjectType, Schema, String
from graphene.relay.connection import Connection, ConnectionField, PageInfo
from graphene.relay.node import Node
class LetterConnection(Connection):

    class Meta:
        node = Letter