import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('mysql', sqla_exc.OperationalError, ".*\\(1305,\\s+\\'SAVEPOINT\\s+(.+)\\s+does not exist\\'\\)")
def _raise_savepoints_as_dberrors(error, match, engine_name, is_disconnect):
    raise exception.DBError(error)