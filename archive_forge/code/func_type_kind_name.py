from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def type_kind_name(type_: GraphQLNamedType) -> str:
    if is_scalar_type(type_):
        return 'a Scalar type'
    if is_object_type(type_):
        return 'an Object type'
    if is_interface_type(type_):
        return 'an Interface type'
    if is_union_type(type_):
        return 'a Union type'
    if is_enum_type(type_):
        return 'an Enum type'
    if is_input_object_type(type_):
        return 'an Input type'
    raise TypeError(f'Unexpected type {inspect(type)}')