from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
class GraphQLWrappingType(GraphQLType, Generic[GT]):
    """Base class for all GraphQL wrapping types"""
    of_type: GT

    def __init__(self, type_: GT) -> None:
        if not is_type(type_):
            raise TypeError(f'Can only create a wrapper for a GraphQLType, but got: {type_}.')
        self.of_type = type_

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.of_type!r}>'