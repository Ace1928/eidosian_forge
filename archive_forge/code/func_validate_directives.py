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
def validate_directives(self) -> None:
    directives = self.schema.directives
    for directive in directives:
        if not is_directive(directive):
            self.report_error(f'Expected directive but got: {inspect(directive)}.', getattr(directive, 'ast_node', None))
            continue
        self.validate_name(directive)
        for arg_name, arg in directive.args.items():
            self.validate_name(arg, arg_name)
            if not is_input_type(arg.type):
                self.report_error(f'The type of @{directive.name}({arg_name}:) must be Input Type but got: {inspect(arg.type)}.', arg.ast_node)
            if is_required_argument(arg) and arg.deprecation_reason is not None:
                self.report_error(f'Required argument @{directive.name}({arg_name}:) cannot be deprecated.', [get_deprecated_directive_node(arg.ast_node), arg.ast_node and arg.ast_node.type])