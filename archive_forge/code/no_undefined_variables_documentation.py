from typing import Any, Set
from ...error import GraphQLError
from ...language import OperationDefinitionNode, VariableDefinitionNode
from . import ValidationContext, ValidationRule
No undefined variables

    A GraphQL operation is only valid if all variables encountered, both directly and
    via fragment spreads, are defined by that operation.

    See https://spec.graphql.org/draft/#sec-All-Variable-Uses-Defined
    