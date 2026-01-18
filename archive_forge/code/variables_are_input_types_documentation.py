from typing import Any
from ...error import GraphQLError
from ...language import VariableDefinitionNode, print_ast
from ...type import is_input_type
from ...utilities import type_from_ast
from . import ValidationRule
Variables are input types

    A GraphQL operation is only valid if all the variables it defines are of input types
    (scalar, enum, or input object).

    See https://spec.graphql.org/draft/#sec-Variables-Are-Input-Types
    