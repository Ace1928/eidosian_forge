from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node
class BasePhoto(Interface):
    width = Int(description='The width of the photo in pixels')