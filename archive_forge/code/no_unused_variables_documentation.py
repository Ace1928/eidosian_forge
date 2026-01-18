from typing import Any, List, Set
from ...error import GraphQLError
from ...language import OperationDefinitionNode, VariableDefinitionNode
from . import ValidationContext, ValidationRule
No unused variables

    A GraphQL operation is only valid if all variables defined by an operation are used,
    either directly or within a spread fragment.

    See https://spec.graphql.org/draft/#sec-All-Variables-Used
    