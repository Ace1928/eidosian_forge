import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@staticmethod
def _extract_member_target_data(member_target_type, member_target):
    """Build some useful target data.

        :param member_target_type: what type of target, e.g. 'user'
        :type member_target_type: str or None
        :param member_target: reference of the target data
        :type member_target: dict or None
        :returns: constructed target dict or empty dict
        :rtype: dict
        """
    ret_dict = {}
    if member_target is not None and member_target_type is None or (member_target is None and member_target_type is not None):
        LOG.warning('RBAC: Unknown target type or target reference. Rejecting as unauthorized. (member_target_type=%(target_type)r, member_target=%(target_ref)r)', {'target_type': member_target_type, 'target_ref': member_target})
        return ret_dict
    if member_target is not None and member_target_type is not None:
        ret_dict['target'] = {member_target_type: member_target}
    elif flask.request.endpoint:
        resource = flask.current_app.view_functions[flask.request.endpoint].view_class
        try:
            member_name = getattr(resource, 'member_key', None)
        except ValueError:
            member_name = None
        func = getattr(resource, 'get_member_from_driver', None)
        if member_name is not None and callable(func):
            key = '%s_id' % member_name
            if key in (flask.request.view_args or {}):
                ret_dict['target'] = {member_name: func(flask.request.view_args[key])}
    return ret_dict