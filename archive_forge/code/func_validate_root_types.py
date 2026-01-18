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
def validate_root_types(self) -> None:
    schema = self.schema
    query_type = schema.query_type
    if not query_type:
        self.report_error('Query root type must be provided.', schema.ast_node)
    elif not is_object_type(query_type):
        self.report_error(f'Query root type must be Object type, it cannot be {query_type}.', get_operation_type_node(schema, OperationType.QUERY) or query_type.ast_node)
    mutation_type = schema.mutation_type
    if mutation_type and (not is_object_type(mutation_type)):
        self.report_error(f'Mutation root type must be Object type if provided, it cannot be {mutation_type}.', get_operation_type_node(schema, OperationType.MUTATION) or mutation_type.ast_node)
    subscription_type = schema.subscription_type
    if subscription_type and (not is_object_type(subscription_type)):
        self.report_error(f'Subscription root type must be Object type if provided, it cannot be {subscription_type}.', get_operation_type_node(schema, OperationType.SUBSCRIPTION) or subscription_type.ast_node)