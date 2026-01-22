from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
class ExecutionResult(object):
    """The result of execution. `data` is the result of executing the
    query, `errors` is null if no errors occurred, and is a
    non-empty array if an error occurred."""
    __slots__ = ('data', 'errors', 'invalid')

    def __init__(self, data=None, errors=None, invalid=False):
        self.data = data
        self.errors = errors
        if invalid:
            assert data is None
        self.invalid = invalid

    def __eq__(self, other):
        return self is other or (isinstance(other, ExecutionResult) and self.data == other.data and (self.errors == other.errors) and (self.invalid == other.invalid))