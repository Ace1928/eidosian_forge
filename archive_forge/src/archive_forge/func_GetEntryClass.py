import collections.abc
import copy
import pickle
from typing import (
def GetEntryClass(self) -> Any:
    return self._entry_descriptor._concrete_class