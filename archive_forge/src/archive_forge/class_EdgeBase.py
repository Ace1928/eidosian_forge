import re
from collections.abc import Iterable
from functools import partial
from typing import Type
from graphql_relay import connection_from_array
from ..types import Boolean, Enum, Int, Interface, List, NonNull, Scalar, String, Union
from ..types.field import Field
from ..types.objecttype import ObjectType, ObjectTypeOptions
from ..utils.thenables import maybe_thenable
from .node import is_node, AbstractNode
class EdgeBase:
    node = Field(NonNull(_node) if strict_types else _node, description='The item at the end of the edge')
    cursor = String(required=True, description='A cursor for use in pagination')