from typing import Any, Dict, Optional, Union
from ...error import GraphQLError
from ...language import (
from ...type import GraphQLObjectType
from . import SDLValidationContext, SDLValidationRule
Unique operation types

    A GraphQL document is only valid if it has only one type per operation.
    