from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
class CommaExpansion(ExpressionExpansion):
    """
    Label Expansion with Comma-Prefix {,var}.

    Non-standard extension to support partial expansions.
    """
    operator = ','
    output_prefix = ','

    def __init__(self, variables: str) -> None:
        super().__init__(variables[1:])

    def _expand_var(self, variable: Variable, value: Any) -> str | None:
        """Expand a single variable."""
        return self._encode_var(variable, self._uri_encode_name(variable.name), value, delim='.' if variable.explode else ',')