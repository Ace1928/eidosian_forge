import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
def add_placeholder(self, alias_id: Hashable, placeholder: Hashable) -> None:
    if alias_id in self._alias_id_to_placeholder:
        raise KeyError(f'alias id: {alias_id} is already stored in this instance of placeholder context.')
    self._alias_id_to_placeholder[alias_id] = placeholder