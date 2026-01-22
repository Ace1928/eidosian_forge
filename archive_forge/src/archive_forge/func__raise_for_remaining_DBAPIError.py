import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('*', sqla_exc.DBAPIError, '.*')
def _raise_for_remaining_DBAPIError(error, match, engine_name, is_disconnect):
    """Filter for remaining DBAPIErrors.

    Filter for remaining DBAPIErrors and wrap if they represent
    a disconnect error.
    """
    if is_disconnect:
        raise exception.DBConnectionError(error)
    else:
        LOG.warning('DBAPIError exception wrapped.', exc_info=True)
        raise exception.DBError(error)