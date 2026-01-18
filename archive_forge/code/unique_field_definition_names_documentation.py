from collections import defaultdict
from typing import Any, Dict
from ...error import GraphQLError
from ...language import NameNode, ObjectTypeDefinitionNode, VisitorAction, SKIP
from ...type import is_object_type, is_interface_type, is_input_object_type
from . import SDLValidationContext, SDLValidationRule
Unique field definition names

    A GraphQL complex type is only valid if all its fields are uniquely named.
    