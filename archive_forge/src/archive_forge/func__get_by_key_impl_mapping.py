from __future__ import annotations
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
def _get_by_key_impl_mapping(self, key: str) -> Any:
    try:
        return self._data[self._key_to_index[key]]
    except KeyError:
        pass
    self._parent._key_not_found(key, False)