from typing import Any, Union, cast
from ...error import GraphQLError
from ...language import (
from . import ASTValidationRule
Executable definitions

    A GraphQL document is only valid for execution if all definitions are either
    operation or fragment definitions.

    See https://spec.graphql.org/draft/#sec-Executable-Definitions
    