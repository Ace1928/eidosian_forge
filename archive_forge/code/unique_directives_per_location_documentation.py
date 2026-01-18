from collections import defaultdict
from typing import Any, Dict, List, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import specified_directives
from . import ASTValidationRule, SDLValidationContext, ValidationContext
Unique directive names per location

    A GraphQL document is only valid if all non-repeatable directives at a given
    location are uniquely named.

    See https://spec.graphql.org/draft/#sec-Directives-Are-Unique-Per-Location
    