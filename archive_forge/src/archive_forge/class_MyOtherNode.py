import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
class MyOtherNode(SharedNodeFields, ObjectType):
    extra_field = String()

    class Meta:
        interfaces = (Node,)

    def resolve_extra_field(self, *_):
        return 'extra field info.'

    @staticmethod
    def get_node(info, id):
        return MyOtherNode(shared=str(id))