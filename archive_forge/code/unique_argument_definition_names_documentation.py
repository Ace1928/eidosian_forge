from operator import attrgetter
from typing import Any, Collection
from ...error import GraphQLError
from ...language import (
from ...pyutils import group_by
from . import SDLValidationRule
Unique argument definition names

    A GraphQL Object or Interface type is only valid if all its fields have uniquely
    named arguments.
    A GraphQL Directive is only valid if all its arguments are uniquely named.

    See https://spec.graphql.org/draft/#sec-Argument-Uniqueness
    