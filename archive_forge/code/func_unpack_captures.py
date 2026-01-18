import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def unpack_captures(self, captures) -> List[core.Tensor]:
    """Unpacks captures to flat tensors."""
    flat = []
    for v, t in zip(captures, self.captures.values()):
        flat.extend(t._to_tensors(v))
    if len(flat) != len(self.flat_captures):
        raise TypeError(f'Flattening captures {captures} with type {self!r} produced {len(flat)} tensors instead of {len(self.flat_captures)}')
    return flat