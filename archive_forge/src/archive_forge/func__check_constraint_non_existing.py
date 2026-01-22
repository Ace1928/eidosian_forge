import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('postgresql', sqla_exc.ProgrammingError, '.* constraint \\"(?P<constraint>.+)\\" of relation "(?P<relation>.+)" does not exist')
@filters('mysql', sqla_exc.InternalError, ".*1091,.*Can't DROP (?:FOREIGN KEY )?['`](?P<constraint>.+)['`]; check that .* exists")
@filters('mysql', sqla_exc.OperationalError, ".*1091,.*Can't DROP (?:FOREIGN KEY )?['`](?P<constraint>.+)['`]; check that .* exists")
@filters('mysql', sqla_exc.InternalError, ".*1025,.*Error on rename of '.+/(?P<relation>.+)' to ")
def _check_constraint_non_existing(programming_error, match, engine_name, is_disconnect):
    """Filter for constraint non existing errors."""
    try:
        relation = match.group('relation')
    except IndexError:
        relation = None
    try:
        constraint = match.group('constraint')
    except IndexError:
        constraint = None
    raise exception.DBNonExistentConstraint(relation, constraint, programming_error)