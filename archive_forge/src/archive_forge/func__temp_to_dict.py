import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def _temp_to_dict(self, *args, **kwargs):
    _replace_to_dict(cls, '_lazyclasses_to_dict')
    return self._lazyclasses_to_dict(*args, **kwargs)