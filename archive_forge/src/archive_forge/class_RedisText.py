from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
class RedisText(RedisFilterField):
    """RedisFilterField representing a text field in a Redis index."""
    OPERATORS: Dict[RedisFilterOperator, str] = {RedisFilterOperator.EQ: '==', RedisFilterOperator.NE: '!=', RedisFilterOperator.LIKE: '%'}
    OPERATOR_MAP: Dict[RedisFilterOperator, str] = {RedisFilterOperator.EQ: '@%s:("%s")', RedisFilterOperator.NE: '(-@%s:"%s")', RedisFilterOperator.LIKE: '@%s:(%s)'}
    SUPPORTED_VAL_TYPES = (str, type(None))

    @check_operator_misuse
    def __eq__(self, other: str) -> 'RedisFilterExpression':
        """Create a RedisText equality (exact match) filter expression.

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisText
            >>> filter = RedisText("job") == "engineer"
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.EQ)
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: str) -> 'RedisFilterExpression':
        """Create a RedisText inequality filter expression.

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisText
            >>> filter = RedisText("job") != "engineer"
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.NE)
        return RedisFilterExpression(str(self))

    def __mod__(self, other: str) -> 'RedisFilterExpression':
        """Create a RedisText "LIKE" filter expression.

        Args:
            other (str): The text value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisText
            >>> filter = RedisText("job") % "engine*"         # suffix wild card match
            >>> filter = RedisText("job") % "%%engine%%"      # fuzzy match w/ LD
            >>> filter = RedisText("job") % "engineer|doctor" # contains either term
            >>> filter = RedisText("job") % "engineer doctor" # contains both terms
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.LIKE)
        return RedisFilterExpression(str(self))

    def __str__(self) -> str:
        """Return the query syntax for a RedisText filter expression."""
        if not self._value:
            return '*'
        return self.OPERATOR_MAP[self._operator] % (self._field, self._value)