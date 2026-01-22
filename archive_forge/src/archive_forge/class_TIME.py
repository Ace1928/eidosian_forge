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
class TIME(_DateTimeMixin, sqltypes.Time):
    """Represent a Python time object in SQLite using a string.

    The default string storage format is::

        "%(hour)02d:%(minute)02d:%(second)02d.%(microsecond)06d"

    e.g.::

        12:05:57.10558

    The incoming storage format is by default parsed using the
    Python ``time.fromisoformat()`` function.

    .. versionchanged:: 2.0  ``time.fromisoformat()`` is used for default
       time string parsing.

    The storage format can be customized to some degree using the
    ``storage_format`` and ``regexp`` parameters, such as::

        import re
        from sqlalchemy.dialects.sqlite import TIME

        t = TIME(storage_format="%(hour)02d-%(minute)02d-"
                                "%(second)02d-%(microsecond)06d",
                 regexp=re.compile("(\\d+)-(\\d+)-(\\d+)-(?:-(\\d+))?")
        )

    :param storage_format: format string which will be applied to the dict
     with keys hour, minute, second, and microsecond.

    :param regexp: regular expression which will be applied to incoming result
     rows, replacing the use of ``datetime.fromisoformat()`` to parse incoming
     strings. If the regexp contains named groups, the resulting match dict is
     applied to the Python time() constructor as keyword arguments. Otherwise,
     if positional groups are used, the time() constructor is called with
     positional arguments via ``*map(int, match_obj.groups(0))``.

    """
    _storage_format = '%(hour)02d:%(minute)02d:%(second)02d.%(microsecond)06d'

    def __init__(self, *args, **kwargs):
        truncate_microseconds = kwargs.pop('truncate_microseconds', False)
        super().__init__(*args, **kwargs)
        if truncate_microseconds:
            assert 'storage_format' not in kwargs, 'You can specify only one of truncate_microseconds or storage_format.'
            assert 'regexp' not in kwargs, 'You can specify only one of truncate_microseconds or regexp.'
            self._storage_format = '%(hour)02d:%(minute)02d:%(second)02d'

    def bind_processor(self, dialect):
        datetime_time = datetime.time
        format_ = self._storage_format

        def process(value):
            if value is None:
                return None
            elif isinstance(value, datetime_time):
                return format_ % {'hour': value.hour, 'minute': value.minute, 'second': value.second, 'microsecond': value.microsecond}
            else:
                raise TypeError('SQLite Time type only accepts Python time objects as input.')
        return process

    def result_processor(self, dialect, coltype):
        if self._reg:
            return processors.str_to_datetime_processor_factory(self._reg, datetime.time)
        else:
            return processors.str_to_time