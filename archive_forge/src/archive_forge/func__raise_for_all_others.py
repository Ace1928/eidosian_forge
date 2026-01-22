import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('*', Exception, '.*')
def _raise_for_all_others(error, match, engine_name, is_disconnect):
    LOG.warning('DB exception wrapped.', exc_info=True)
    raise exception.DBError(error)