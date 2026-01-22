from ..language.printer import print_ast
from ..type.definition import (GraphQLEnumType, GraphQLInputObjectType,
from ..type.directives import DEFAULT_DEPRECATION_REASON
from .ast_from_value import ast_from_value
def _print_union(type):
    return 'union {} = {}'.format(type.name, ' | '.join((str(t) for t in type.types)))