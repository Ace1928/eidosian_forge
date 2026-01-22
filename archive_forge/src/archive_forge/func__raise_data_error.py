import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('mysql', sqla_exc.OperationalError, '.*(1292|1366).*Incorrect \\w+ value.*')
@filters('mysql', sqla_exc.DataError, '.*1265.*Data truncated for column.*')
@filters('mysql', sqla_exc.DataError, '.*1264.*Out of range value for column.*')
@filters('mysql', sqla_exc.InternalError, '^.*1366.*Incorrect string value:*')
@filters('sqlite', sqla_exc.ProgrammingError, '(?i).*You must not use 8-bit bytestrings*')
@filters('mysql', sqla_exc.DataError, '.*1406.*Data too long for column.*')
def _raise_data_error(error, match, engine_name, is_disconnect):
    """Raise DBDataError exception for different data errors."""
    raise exception.DBDataError(error)