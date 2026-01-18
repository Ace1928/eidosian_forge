from typing import Any
from ...error import GraphQLError
from ...language import SchemaDefinitionNode
from . import SDLValidationRule, SDLValidationContext
Lone Schema definition

    A GraphQL document is only valid if it contains only one schema definition.
    