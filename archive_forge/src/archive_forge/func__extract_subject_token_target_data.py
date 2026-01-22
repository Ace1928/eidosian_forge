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
@classmethod
def _extract_subject_token_target_data(cls):
    ret_dict = {}
    window_seconds = 0
    target = 'token'
    subject_token = flask.request.headers.get('X-Subject-Token')
    access_rules_support = flask.request.headers.get(authorization.ACCESS_RULES_HEADER)
    if subject_token is not None:
        allow_expired = strutils.bool_from_string(flask.request.args.get('allow_expired', False), default=False)
        if allow_expired:
            window_seconds = CONF.token.allow_expired_window
        token = PROVIDER_APIS.token_provider_api.validate_token(subject_token, window_seconds=window_seconds, access_rules_support=access_rules_support)
        ret_dict[target] = {}
        ret_dict[target]['user_id'] = token.user_id
        try:
            user_domain_id = token.user['domain_id']
        except exception.UnexpectedError:
            user_domain_id = None
        if user_domain_id:
            ret_dict[target].setdefault('user', {})
            ret_dict[target]['user'].setdefault('domain', {})
            ret_dict[target]['user']['domain']['id'] = user_domain_id
    return ret_dict