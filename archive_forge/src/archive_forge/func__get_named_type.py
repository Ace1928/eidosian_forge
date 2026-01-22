from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
def _get_named_type(typeName):
    cached_type_def = type_def_cache.get(typeName)
    if cached_type_def:
        return cached_type_def
    existing_type = schema.get_type(typeName)
    if existing_type:
        type_def = extend_type(existing_type)
        type_def_cache[typeName] = type_def
        return type_def
    type_ast = type_definition_map.get(typeName)
    if type_ast:
        type_def = build_type(type_ast)
        type_def_cache[typeName] = type_def
        return type_def