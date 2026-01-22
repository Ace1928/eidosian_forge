from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
class RedisFilterExpression:
    """Logical expression of RedisFilterFields.

    RedisFilterExpressions can be combined using the & and | operators to create
    complex logical expressions that evaluate to the Redis Query language.

    This presents an interface by which users can create complex queries
    without having to know the Redis Query language.

    Filter expressions are not initialized directly. Instead they are built
    by combining RedisFilterFields using the & and | operators.

    Examples:

        >>> from langchain_community.vectorstores.redis import RedisTag, RedisNum
        >>> brand_is_nike = RedisTag("brand") == "nike"
        >>> price_is_under_100 = RedisNum("price") < 100
        >>> filter = brand_is_nike & price_is_under_100
        >>> print(str(filter))
        (@brand:{nike} @price:[-inf (100)])

    """

    def __init__(self, _filter: Optional[str]=None, operator: Optional[RedisFilterOperator]=None, left: Optional['RedisFilterExpression']=None, right: Optional['RedisFilterExpression']=None):
        self._filter = _filter
        self._operator = operator
        self._left = left
        self._right = right

    def __and__(self, other: 'RedisFilterExpression') -> 'RedisFilterExpression':
        return RedisFilterExpression(operator=RedisFilterOperator.AND, left=self, right=other)

    def __or__(self, other: 'RedisFilterExpression') -> 'RedisFilterExpression':
        return RedisFilterExpression(operator=RedisFilterOperator.OR, left=self, right=other)

    @staticmethod
    def format_expression(left: 'RedisFilterExpression', right: 'RedisFilterExpression', operator_str: str) -> str:
        _left, _right = (str(left), str(right))
        if _left == _right == '*':
            return _left
        if _left == '*' != _right:
            return _right
        if _right == '*' != _left:
            return _left
        return f'({_left}{operator_str}{_right})'

    def __str__(self) -> str:
        if not self._filter and (not self._operator):
            raise ValueError('Improperly initialized RedisFilterExpression')
        if self._operator:
            if not isinstance(self._left, RedisFilterExpression) or not isinstance(self._right, RedisFilterExpression):
                raise TypeError('Improper combination of filters.Both left and right should be type FilterExpression')
            operator_str = ' | ' if self._operator == RedisFilterOperator.OR else ' '
            return self.format_expression(self._left, self._right, operator_str)
        if not self._filter:
            raise ValueError('Improperly initialized RedisFilterExpression')
        return self._filter