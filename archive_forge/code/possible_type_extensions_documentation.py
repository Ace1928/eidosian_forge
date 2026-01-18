import re
from functools import partial
from typing import Any, Optional
from ...error import GraphQLError
from ...language import TypeDefinitionNode, TypeExtensionNode
from ...pyutils import did_you_mean, inspect, suggestion_list
from ...type import (
from . import SDLValidationContext, SDLValidationRule
Possible type extension

    A type extension is only valid if the type is defined and has the same kind.
    