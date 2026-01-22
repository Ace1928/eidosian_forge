from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
def anyof(self, *strings: str) -> str:
    for s in strings:
        if self.static_b(s):
            return s
    else:
        raise nomatch