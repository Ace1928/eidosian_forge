from typing import Any, Dict
from ...error import GraphQLError
from ...language import DirectiveDefinitionNode, NameNode, VisitorAction, SKIP
from . import SDLValidationContext, SDLValidationRule
Unique directive names

    A GraphQL document is only valid if all defined directives have unique names.
    