from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
class GraphQLResolveInfo(NamedTuple):
    """Collection of information passed to the resolvers.

    This is always passed as the first argument to the resolvers.

    Note that contrary to the JavaScript implementation, the context (commonly used to
    represent an authenticated user, or request-specific caches) is included here and
    not passed as an additional argument.
    """
    field_name: str
    field_nodes: List[FieldNode]
    return_type: 'GraphQLOutputType'
    parent_type: 'GraphQLObjectType'
    path: Path
    schema: 'GraphQLSchema'
    fragments: Dict[str, FragmentDefinitionNode]
    root_value: Any
    operation: OperationDefinitionNode
    variable_values: Dict[str, Any]
    context: Any
    is_awaitable: Callable[[Any], bool]