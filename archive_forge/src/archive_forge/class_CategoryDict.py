import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
@dataclasses.dataclass
class CategoryDict:
    _values: DefaultDict[int, CategoryElement] = dataclasses.field(default_factory=lambda: collections.defaultdict(CategoryElement))

    def set_by_id(self, key: TensorKey, category: Category) -> None:
        self._values[key.id].by_id = category
        self._values[key.id]._by_id_keyset.add(key)

    def set_by_key(self, key: TensorKey, category: Category) -> None:
        self._values[key.id].by_key[key] = category

    def set_by_version(self, key: TensorKey, version: int, category: Category) -> None:
        self._values[key.id].by_version[key, version] = category

    def setdefault_by_version(self, key: TensorKey, version: int, category: Category) -> None:
        self._values[key.id].by_version.setdefault((key, version), category)

    def get(self, key: Key, version: int) -> Optional[Category]:
        if isinstance(key, Key) and (not isinstance(key, TensorKey)):
            return None
        element = self._values[key.id]
        return element.by_id or element.by_key.get(key, None) or element.by_version.get((key, version), None)