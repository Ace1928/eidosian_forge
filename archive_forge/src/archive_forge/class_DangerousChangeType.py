from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
class DangerousChangeType(Enum):
    VALUE_ADDED_TO_ENUM = 60
    TYPE_ADDED_TO_UNION = 61
    OPTIONAL_INPUT_FIELD_ADDED = 62
    OPTIONAL_ARG_ADDED = 63
    IMPLEMENTED_INTERFACE_ADDED = 64
    ARG_DEFAULT_VALUE_CHANGE = 65