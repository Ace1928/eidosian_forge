from typing import cast, Any, Dict, List, Union
from ...error import GraphQLError
from ...language import (
from ...pyutils import did_you_mean, suggestion_list
from ...type import specified_directives
from . import ASTValidationRule, SDLValidationContext, ValidationContext
Known argument names

    A GraphQL field is only valid if all supplied arguments are defined by that field.

    See https://spec.graphql.org/draft/#sec-Argument-Names
    See https://spec.graphql.org/draft/#sec-Directives-Are-In-Valid-Locations
    