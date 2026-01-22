from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
class DATE(_DateTimeMixin, sqltypes.Date):
    """Represent a Python date object in SQLite using a string.

    The default string storage format is::

        "%(year)04d-%(month)02d-%(day)02d"

    e.g.::

        2011-03-15

    The incoming storage format is by default parsed using the
    Python ``date.fromisoformat()`` function.

    .. versionchanged:: 2.0  ``date.fromisoformat()`` is used for default
       date string parsing.


    The storage format can be customized to some degree using the
    ``storage_format`` and ``regexp`` parameters, such as::

        import re
        from sqlalchemy.dialects.sqlite import DATE

        d = DATE(
                storage_format="%(month)02d/%(day)02d/%(year)04d",
                regexp=re.compile("(?P<month>\\d+)/(?P<day>\\d+)/(?P<year>\\d+)")
            )

    :param storage_format: format string which will be applied to the
     dict with keys year, month, and day.

    :param regexp: regular expression which will be applied to
     incoming result rows, replacing the use of ``date.fromisoformat()`` to
     parse incoming strings. If the regexp contains named groups, the resulting
     match dict is applied to the Python date() constructor as keyword
     arguments. Otherwise, if positional groups are used, the date()
     constructor is called with positional arguments via
     ``*map(int, match_obj.groups(0))``.

    """
    _storage_format = '%(year)04d-%(month)02d-%(day)02d'

    def bind_processor(self, dialect):
        datetime_date = datetime.date
        format_ = self._storage_format

        def process(value):
            if value is None:
                return None
            elif isinstance(value, datetime_date):
                return format_ % {'year': value.year, 'month': value.month, 'day': value.day}
            else:
                raise TypeError('SQLite Date type only accepts Python date objects as input.')
        return process

    def result_processor(self, dialect, coltype):
        if self._reg:
            return processors.str_to_datetime_processor_factory(self._reg, datetime.date)
        else:
            return processors.str_to_date