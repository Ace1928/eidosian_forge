import datetime
import functools
import pytz
from oslo_db import exception as db_exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import models
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from osprofiler import opts as profiler
import osprofiler.sqlalchemy
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.orm.attributes import flag_modified, InstrumentedAttribute
from sqlalchemy import types as sql_types
from keystone.common import driver_hints
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class DateTimeInt(sql_types.TypeDecorator):
    """A column that automatically converts a datetime object to an Int.

    Keystone relies on accurate (sub-second) datetime objects. In some cases
    the RDBMS drop sub-second accuracy (some versions of MySQL). This field
    automatically converts the value to an INT when storing the data and
    back to a datetime object when it is loaded from the database.

    NOTE: Any datetime object that has timezone data will be converted to UTC.
          Any datetime object that has no timezone data will be assumed to be
          UTC and loaded from the DB as such.
    """
    impl = sql.BigInteger
    epoch = datetime.datetime.fromtimestamp(0, tz=pytz.UTC)
    cache_ok = True
    'This type is safe to cache.'

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, datetime.datetime):
                raise ValueError(_('Programming Error: value to be stored must be a datetime object.'))
            value = timeutils.normalize_time(value)
            value = value.replace(tzinfo=pytz.UTC)
            return int((value - self.epoch).total_seconds() * 1000000)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        else:
            value = float(value) / 1000000
            dt_obj = datetime.datetime.fromtimestamp(value, tz=pytz.UTC)
            return timeutils.normalize_time(dt_obj)