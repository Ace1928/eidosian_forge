import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('mysql', sqla_exc.InternalError, '.*1049,.*Unknown database \'(?P<database>.+)\'\\"')
@filters('mysql', sqla_exc.OperationalError, '.*1049,.*Unknown database \'(?P<database>.+)\'\\"')
@filters('postgresql', sqla_exc.OperationalError, '.*database \\"(?P<database>.+)\\" does not exist')
@filters('sqlite', sqla_exc.OperationalError, '.*unable to open database file.*')
def _check_database_non_existing(error, match, engine_name, is_disconnect):
    try:
        database = match.group('database')
    except IndexError:
        database = None
    raise exception.DBNonExistentDatabase(database, error)