import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('mysql', sqla_exc.OperationalError, '.*\\(.*(?:2002|2003|2006|2013|1047)')
@filters('mysql', sqla_exc.InternalError, '.*\\(.*(?:1927)')
@filters('mysql', sqla_exc.InternalError, '.*Packet sequence number wrong')
@filters('postgresql', sqla_exc.OperationalError, '.*could not connect to server')
def _is_db_connection_error(operational_error, match, engine_name, is_disconnect):
    """Detect the exception as indicating a recoverable error on connect."""
    raise exception.DBConnectionError(operational_error)