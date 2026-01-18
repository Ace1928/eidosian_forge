from typing import Any, Dict
from ...error import GraphQLError
from ...language import NameNode, TypeDefinitionNode, VisitorAction, SKIP
from . import SDLValidationContext, SDLValidationRule
Unique type names

    A GraphQL document is only valid if all defined types have unique names.
    