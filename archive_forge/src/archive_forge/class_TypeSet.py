from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
class TypeSet(Dict[GraphQLNamedType, None]):
    """An ordered set of types that can be collected starting from initial types."""

    @classmethod
    def with_initial_types(cls, types: Collection[GraphQLType]) -> 'TypeSet':
        return cast(TypeSet, super().fromkeys(types))

    def collect_referenced_types(self, type_: GraphQLType) -> None:
        """Recursive function supplementing the type starting from an initial type."""
        named_type = get_named_type(type_)
        if named_type in self:
            return
        self[named_type] = None
        collect_referenced_types = self.collect_referenced_types
        if is_union_type(named_type):
            named_type = cast(GraphQLUnionType, named_type)
            for member_type in named_type.types:
                collect_referenced_types(member_type)
        elif is_object_type(named_type) or is_interface_type(named_type):
            named_type = cast(Union[GraphQLObjectType, GraphQLInterfaceType], named_type)
            for interface_type in named_type.interfaces:
                collect_referenced_types(interface_type)
            for field in named_type.fields.values():
                collect_referenced_types(field.type)
                for arg in field.args.values():
                    collect_referenced_types(arg.type)
        elif is_input_object_type(named_type):
            named_type = cast(GraphQLInputObjectType, named_type)
            for field in named_type.fields.values():
                collect_referenced_types(field.type)