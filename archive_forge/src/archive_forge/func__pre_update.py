import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable
def _pre_update(self, op: str, need_reindex: bool=True) -> None:
    if self.readonly:
        raise InvalidOperationError('This dict is readonly')
    self._need_reindex = need_reindex