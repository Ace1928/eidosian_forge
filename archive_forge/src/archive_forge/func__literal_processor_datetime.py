from __future__ import annotations
import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR
def _literal_processor_datetime(self, dialect):

    def process(value):
        if getattr(value, 'microsecond', None):
            value = f"TO_TIMESTAMP('{value.isoformat().replace('T', ' ')}', 'YYYY-MM-DD HH24:MI:SS.FF')"
        else:
            value = f"TO_DATE('{value.isoformat().replace('T', ' ')}', 'YYYY-MM-DD HH24:MI:SS')"
        return value
    return process