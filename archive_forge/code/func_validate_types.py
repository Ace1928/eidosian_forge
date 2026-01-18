from operator import attrgetter, itemgetter
from typing import (
from ..error import GraphQLError
from ..pyutils import inspect
from ..language import (
from .definition import (
from ..utilities.type_comparators import is_equal_type, is_type_sub_type_of
from .directives import is_directive, GraphQLDeprecatedDirective
from .introspection import is_introspection_type
from .schema import GraphQLSchema, assert_schema
def validate_types(self) -> None:
    validate_input_object_circular_refs = InputObjectCircularRefsValidator(self)
    for type_ in self.schema.type_map.values():
        if not is_named_type(type_):
            self.report_error(f'Expected GraphQL named type but got: {inspect(type_)}.', type_.ast_node if is_named_type(type_) else None)
            continue
        if not is_introspection_type(type_):
            self.validate_name(type_)
        if is_object_type(type_):
            type_ = cast(GraphQLObjectType, type_)
            self.validate_fields(type_)
            self.validate_interfaces(type_)
        elif is_interface_type(type_):
            type_ = cast(GraphQLInterfaceType, type_)
            self.validate_fields(type_)
            self.validate_interfaces(type_)
        elif is_union_type(type_):
            type_ = cast(GraphQLUnionType, type_)
            self.validate_union_members(type_)
        elif is_enum_type(type_):
            type_ = cast(GraphQLEnumType, type_)
            self.validate_enum_values(type_)
        elif is_input_object_type(type_):
            type_ = cast(GraphQLInputObjectType, type_)
            self.validate_input_fields(type_)
            validate_input_object_circular_refs(type_)