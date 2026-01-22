from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_input_value(name, arg):
    if arg.default_value is not None:
        default_value = ' = ' + print_ast(ast_from_value(arg.default_value, arg.type))
    else:
        default_value = ''
    return '{}: {}{}'.format(name, arg.type, default_value)