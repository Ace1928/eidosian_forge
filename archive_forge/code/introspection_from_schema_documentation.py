from typing import cast
from ..error import GraphQLError
from ..language import parse
from ..type import GraphQLSchema
from .get_introspection_query import get_introspection_query, IntrospectionQuery
Build an IntrospectionQuery from a GraphQLSchema

    IntrospectionQuery is useful for utilities that care about type and field
    relationships, but do not need to traverse through those relationships.

    This is the inverse of build_client_schema. The primary use case is outside of the
    server context, for instance when doing schema comparisons.
    