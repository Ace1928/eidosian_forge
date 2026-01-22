from oslo_config import cfg
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_log import log as logging
from oslo_utils import units
from glance.common import exception
from glance.db.sqlalchemy import api as db
from glance.i18n import _LE
def _enforce_one(context, project_id, quota_name, get_value_fn, delta=0):
    """Helper method to enforce a single named quota value.

    :param context: The RequestContext
    :param project_id: The project_id of the tenant being checked
    :param quota_name: One of the quota names defined above
    :param get_value_fn: A function that will be called with no arguments to
                         return the numerical value representing current usage.
    :param delta: The amount of resource being requested (to be added to the
                  current usage before determining if over-quota).
    :raises: exception.LimitExceeded if the current usage is over the defined
             limit.
    :returns: None if the tenant is not currently over their quota.
    """
    return _enforce_some(context, project_id, {quota_name: get_value_fn}, {quota_name: delta})