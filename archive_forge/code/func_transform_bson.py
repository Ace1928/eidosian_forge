from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
@abc.abstractmethod
def transform_bson(self, value: Any) -> Any:
    """Convert the given BSON value into our own type."""