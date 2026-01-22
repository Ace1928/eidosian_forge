import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
def _check_and_fill_registered_limit_id(self, limit):
    hints = driver_hints.Hints()
    limit_copy = copy.deepcopy(limit)
    hints.add_filter('service_id', limit_copy.pop('service_id'))
    hints.add_filter('resource_name', limit_copy.pop('resource_name'))
    hints.add_filter('region_id', limit_copy.pop('region_id', None))
    with sql.session_for_read() as session:
        registered_limits = session.query(RegisteredLimitModel)
        registered_limits = sql.filter_limit_query(RegisteredLimitModel, registered_limits, hints)
    reg_limits = registered_limits.all()
    if not reg_limits:
        raise exception.NoLimitReference
    limit_copy['registered_limit_id'] = reg_limits[0]['id']
    return limit_copy