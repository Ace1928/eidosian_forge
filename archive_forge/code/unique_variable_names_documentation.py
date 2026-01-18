from operator import attrgetter
from typing import Any
from ...error import GraphQLError
from ...language import OperationDefinitionNode
from ...pyutils import group_by
from . import ASTValidationRule
Unique variable names

    A GraphQL operation is only valid if all its variables are uniquely named.
    