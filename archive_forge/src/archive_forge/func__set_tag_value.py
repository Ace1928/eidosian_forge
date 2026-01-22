from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
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