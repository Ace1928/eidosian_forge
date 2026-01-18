from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
@classmethod
def with_initial_types(cls, types: Collection[GraphQLType]) -> 'TypeSet':
    return cast(TypeSet, super().fromkeys(types))