from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_enum(type):
    return 'enum {} {{\n{}\n}}'.format(type.name, '\n'.join(('  ' + v.name + _print_deprecated(v) for v in type.values)))