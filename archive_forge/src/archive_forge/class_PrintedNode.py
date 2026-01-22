from typing import Any, Collection, Optional
from ..language.ast import Node, OperationType
from .block_string import print_block_string
from .print_string import print_string
from .visitor import visit, Visitor
class PrintedNode:
    """A union type for all nodes that have been processed by the printer."""
    alias: str
    arguments: Strings
    block: bool
    default_value: str
    definitions: Strings
    description: str
    directives: str
    fields: Strings
    interfaces: Strings
    locations: Strings
    name: str
    operation: OperationType
    operation_types: Strings
    repeatable: bool
    selection_set: str
    selections: Strings
    type: str
    type_condition: str
    types: Strings
    value: str
    values: Strings
    variable: str
    variable_definitions: Strings