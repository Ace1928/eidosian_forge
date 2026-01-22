from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from langchain_community.utilities.redis import TokenEscaper
class RedisFilter:
    """Collection of RedisFilterFields."""

    @staticmethod
    def text(field: str) -> 'RedisText':
        return RedisText(field)

    @staticmethod
    def num(field: str) -> 'RedisNum':
        return RedisNum(field)

    @staticmethod
    def tag(field: str) -> 'RedisTag':
        return RedisTag(field)