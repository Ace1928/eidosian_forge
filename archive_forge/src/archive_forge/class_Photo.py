from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node
class Photo(ObjectType):

    class Meta:
        interfaces = [CustomNode, BasePhoto]