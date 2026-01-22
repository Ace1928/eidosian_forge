import io
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, Union
import pickle  # noqa: F401
import ray
from ray.dag.base import DAGNodeBase
def _replace_index(self, i: int) -> SourceType:
    return self._replace_table[self._found[i]]