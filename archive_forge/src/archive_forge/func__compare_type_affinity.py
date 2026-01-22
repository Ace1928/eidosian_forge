from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
def _compare_type_affinity(self, other):
    return other._type_affinity in (sqltypes.DateTime, sqltypes.Date)