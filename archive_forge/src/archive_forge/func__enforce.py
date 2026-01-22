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
def _enforce(self, credentials, action, target, do_raise=True):
    """Verify that the action is valid on the target in this context.

        This method is for cases that exceed the base enforcer
        functionality (notably for compatibility with `@protected` style
        decorators.

        :param credentials: user credentials
        :param action: string representing the action to be checked, which
                       should be colon separated for clarity.
        :param target: dictionary representing the object of the action for
                       object creation this should be a dictionary
                       representing the location of the object e.g.
                       {'project_id': object.project_id}
        :raises keystone.exception.Forbidden: If verification fails.

        Actions should be colon separated for clarity. For example:

        * identity:list_users
        """
    extra = {}
    if do_raise:
        extra.update(exc=exception.ForbiddenAction, action=action, do_raise=do_raise)
    try:
        result = self._enforcer.enforce(rule=action, target=target, creds=credentials, **extra)
        self._check_deprecated_rule(action)
        return result
    except common_policy.InvalidScope:
        raise exception.ForbiddenAction(action=action)