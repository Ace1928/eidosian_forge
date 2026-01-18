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
def validate_union_members(self, union: GraphQLUnionType) -> None:
    member_types = union.types
    if not member_types:
        self.report_error(f'Union type {union.name} must define one or more member types.', [union.ast_node, *union.extension_ast_nodes])
    included_type_names: Set[str] = set()
    for member_type in member_types:
        if is_object_type(member_type):
            if member_type.name in included_type_names:
                self.report_error(f'Union type {union.name} can only include type {member_type.name} once.', get_union_member_type_nodes(union, member_type.name))
            else:
                included_type_names.add(member_type.name)
        else:
            self.report_error(f'Union type {union.name} can only include Object types, it cannot include {inspect(member_type)}.', get_union_member_type_nodes(union, str(member_type)))