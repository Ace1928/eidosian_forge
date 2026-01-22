from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
class RedisNum(RedisFilterField):
    """RedisFilterField representing a numeric field in a Redis index."""
    OPERATORS: Dict[RedisFilterOperator, str] = {RedisFilterOperator.EQ: '==', RedisFilterOperator.NE: '!=', RedisFilterOperator.LT: '<', RedisFilterOperator.GT: '>', RedisFilterOperator.LE: '<=', RedisFilterOperator.GE: '>='}
    OPERATOR_MAP: Dict[RedisFilterOperator, str] = {RedisFilterOperator.EQ: '@%s:[%s %s]', RedisFilterOperator.NE: '(-@%s:[%s %s])', RedisFilterOperator.GT: '@%s:[(%s +inf]', RedisFilterOperator.LT: '@%s:[-inf (%s]', RedisFilterOperator.GE: '@%s:[%s +inf]', RedisFilterOperator.LE: '@%s:[-inf %s]'}
    SUPPORTED_VAL_TYPES = (int, float, type(None))

    def __str__(self) -> str:
        """Return the query syntax for a RedisNum filter expression."""
        if self._value is None:
            return '*'
        if self._operator == RedisFilterOperator.EQ or self._operator == RedisFilterOperator.NE:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value, self._value)
        else:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)

    @check_operator_misuse
    def __eq__(self, other: Union[int, float]) -> 'RedisFilterExpression':
        """Create a Numeric equality filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("zipcode") == 90210
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.EQ)
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: Union[int, float]) -> 'RedisFilterExpression':
        """Create a Numeric inequality filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("zipcode") != 90210
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.NE)
        return RedisFilterExpression(str(self))

    def __gt__(self, other: Union[int, float]) -> 'RedisFilterExpression':
        """Create a Numeric greater than filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") > 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.GT)
        return RedisFilterExpression(str(self))

    def __lt__(self, other: Union[int, float]) -> 'RedisFilterExpression':
        """Create a Numeric less than filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") < 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.LT)
        return RedisFilterExpression(str(self))

    def __ge__(self, other: Union[int, float]) -> 'RedisFilterExpression':
        """Create a Numeric greater than or equal to filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") >= 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.GE)
        return RedisFilterExpression(str(self))

    def __le__(self, other: Union[int, float]) -> 'RedisFilterExpression':
        """Create a Numeric less than or equal to filter expression.

        Args:
            other (Union[int, float]): The value to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisNum
            >>> filter = RedisNum("age") <= 18
        """
        self._set_value(other, self.SUPPORTED_VAL_TYPES, RedisFilterOperator.LE)
        return RedisFilterExpression(str(self))