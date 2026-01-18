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
def validate_interfaces(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType]) -> None:
    iface_type_names: Set[str] = set()
    for iface in type_.interfaces:
        if not is_interface_type(iface):
            self.report_error(f'Type {type_.name} must only implement Interface types, it cannot implement {inspect(iface)}.', get_all_implements_interface_nodes(type_, iface))
            continue
        if type_ is iface:
            self.report_error(f'Type {type_.name} cannot implement itself because it would create a circular reference.', get_all_implements_interface_nodes(type_, iface))
        if iface.name in iface_type_names:
            self.report_error(f'Type {type_.name} can only implement {iface.name} once.', get_all_implements_interface_nodes(type_, iface))
            continue
        iface_type_names.add(iface.name)
        self.validate_type_implements_ancestors(type_, iface)
        self.validate_type_implements_interface(type_, iface)