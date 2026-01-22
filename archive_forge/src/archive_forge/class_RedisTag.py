from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
class RedisTag(RedisFilterField):
    """RedisFilterField representing a tag in a Redis index."""
    OPERATORS: Dict[RedisFilterOperator, str] = {RedisFilterOperator.EQ: '==', RedisFilterOperator.NE: '!=', RedisFilterOperator.IN: '=='}
    OPERATOR_MAP: Dict[RedisFilterOperator, str] = {RedisFilterOperator.EQ: '@%s:{%s}', RedisFilterOperator.NE: '(-@%s:{%s})', RedisFilterOperator.IN: '@%s:{%s}'}
    SUPPORTED_VAL_TYPES = (list, set, tuple, str, type(None))

    def __init__(self, field: str):
        """Create a RedisTag FilterField.

        Args:
            field (str): The name of the RedisTag field in the index to be queried
                against.
        """
        super().__init__(field)

    def _set_tag_value(self, other: Union[List[str], Set[str], Tuple[str], str], operator: RedisFilterOperator) -> None:
        if isinstance(other, (list, set, tuple)):
            try:
                other = [str(val) for val in other if val]
            except ValueError:
                raise ValueError('All tags within collection must be strings')
        elif not other:
            other = []
        elif isinstance(other, str):
            other = [other]
        self._set_value(other, self.SUPPORTED_VAL_TYPES, operator)

    @check_operator_misuse
    def __eq__(self, other: Union[List[str], Set[str], Tuple[str], str]) -> 'RedisFilterExpression':
        """Create a RedisTag equality filter expression.

        Args:
            other (Union[List[str], Set[str], Tuple[str], str]):
                The tag(s) to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisTag
            >>> filter = RedisTag("brand") == "nike"
        """
        self._set_tag_value(other, RedisFilterOperator.EQ)
        return RedisFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: Union[List[str], Set[str], Tuple[str], str]) -> 'RedisFilterExpression':
        """Create a RedisTag inequality filter expression.

        Args:
            other (Union[List[str], Set[str], Tuple[str], str]):
                The tag(s) to filter on.

        Example:
            >>> from langchain_community.vectorstores.redis import RedisTag
            >>> filter = RedisTag("brand") != "nike"
        """
        self._set_tag_value(other, RedisFilterOperator.NE)
        return RedisFilterExpression(str(self))

    @property
    def _formatted_tag_value(self) -> str:
        return '|'.join([self.escaper.escape(tag) for tag in self._value])

    def __str__(self) -> str:
        """Return the query syntax for a RedisTag filter expression."""
        if not self._value:
            return '*'
        return self.OPERATOR_MAP[self._operator] % (self._field, self._formatted_tag_value)