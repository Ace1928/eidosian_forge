from __future__ import annotations
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable
def to_metric_str(self) -> str:
    return f'cache_memory_bytes{{cache_type="{self.category_name}",cache="{self.cache_name}"}} {self.byte_length}'