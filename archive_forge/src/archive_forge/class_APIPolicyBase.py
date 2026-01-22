from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
class APIPolicyBase(object):

    def __init__(self, context, target=None, enforcer=None):
        self._context = context
        self._target = target or {}
        self.enforcer = enforcer or policy.Enforcer()

    def _enforce(self, rule_name):
        try:
            self.enforcer.enforce(self._context, rule_name, self._target)
        except exception.Forbidden as e:
            raise webob.exc.HTTPForbidden(explanation=str(e))

    def check(self, name, *args):
        """Perform a soft check of a named policy.

        This is used when you need to check if a policy is allowed for the
        given resource, without needing to catch an exception. If the policy
        check requires args, those are accepted here as well.

        :param name: Policy name to check
        :returns: bool indicating if the policy is allowed.
        """
        try:
            getattr(self, name)(*args)
            return True
        except webob.exc.HTTPForbidden:
            return False