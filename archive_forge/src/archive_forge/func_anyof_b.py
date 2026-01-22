from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
def anyof_b(self, *strings: str) -> bool:
    for s in strings:
        if self.static_b(s):
            return True
    else:
        return False