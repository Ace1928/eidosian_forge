from typing import Any
from ...error import GraphQLError
from ...language import DocumentNode, OperationDefinitionNode
from . import ASTValidationContext, ASTValidationRule
Lone anonymous operation

    A GraphQL document is only valid if when it contains an anonymous operation
    (the query short-hand) that it contains only that one operation definition.

    See https://spec.graphql.org/draft/#sec-Lone-Anonymous-Operation
    