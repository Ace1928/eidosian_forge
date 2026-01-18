from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def type_def_named(type_name):
    if type_name in inner_type_map:
        return inner_type_map[type_name]
    if type_name not in ast_map:
        raise Exception('Type "{}" not found in document'.format(type_name))
    inner_type_def = make_schema_def(ast_map[type_name])
    if not inner_type_def:
        raise Exception('Nothing constructed for "{}".'.format(type_name))
    inner_type_map[type_name] = inner_type_def
    return inner_type_def