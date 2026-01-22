from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
class BINARY_FLOAT(sqltypes.Float):
    __visit_name__ = 'BINARY_FLOAT'