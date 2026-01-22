from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
@property
def _formatted_tag_value(self) -> str:
    return '|'.join([self.escaper.escape(tag) for tag in self._value])