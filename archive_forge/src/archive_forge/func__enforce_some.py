from oslo_config import cfg
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_log import log as logging
from oslo_utils import units
from glance.common import exception
from glance.db.sqlalchemy import api as db
from glance.i18n import _LE
def _enforce_some(context, project_id, quota_value_fns, deltas):
    """Helper method to enforce a set of quota values.

    :param context: The RequestContext
    :param project_id: The project_id of the tenant being checked
    :param get_value_fns: A mapping of quota names to functions that will be
                          called with no arguments to return the numerical
                          value representing current usage.
    :param deltas: A mapping of quota names to the amount of resource being
                   requested for each (to be added to the current usage before
                   determining if over-quota).
    :raises: exception.LimitExceeded if the current usage is over the defined
             limit.
    :returns: None if the tenant is not currently over their quota.
    """
    if not CONF.use_keystone_limits:
        return

    def callback(project_id, resource_names):
        return {name: quota_value_fns[name]() for name in resource_names}
    enforcer = limit.Enforcer(callback)
    try:
        enforcer.enforce(project_id, {quota_name: deltas.get(quota_name, 0) for quota_name in quota_value_fns})
    except ol_exc.ProjectOverLimit as e:
        raise exception.LimitExceeded(body=str(e))
    except ol_exc.SessionInitError as e:
        LOG.error(_LE('Failed to initialize oslo_limit, likely due to incorrect or insufficient configuration: %(err)s'), {'err': str(e)})
        raise