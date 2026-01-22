from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from oslo_utils import excutils
from heat.common import exception
from heat.common.i18n import _
from heat import policies
class ResourceEnforcer(Enforcer):

    def __init__(self, default_rule=DEFAULT_RESOURCE_RULES['default'], **kwargs):
        super(ResourceEnforcer, self).__init__(default_rule=default_rule, **kwargs)
        self.log_not_registered = False

    def _enforce(self, context, res_type, scope=None, target=None, is_registered_policy=False):
        try:
            result = super(ResourceEnforcer, self).enforce(context, res_type, scope=scope or 'resource_types', target=target, is_registered_policy=is_registered_policy)
        except policy.PolicyNotRegistered:
            result = True
        except self.exc as ex:
            LOG.info(str(ex))
            raise
        if not result:
            if self.exc:
                raise self.exc(action=res_type)
        return result

    def enforce(self, context, res_type, scope=None, target=None, is_registered_policy=False):
        result = self._enforce(context, res_type, scope, target, is_registered_policy=is_registered_policy)
        if result:
            subparts = res_type.split('::')[:-1]
            subparts.append('*')
            res_type_wc = '::'.join(subparts)
            try:
                return self._enforce(context, res_type_wc, scope, target, is_registered_policy=is_registered_policy)
            except self.exc:
                raise self.exc(action=res_type)
        return result

    def enforce_stack(self, stack, scope=None, target=None, is_registered_policy=False):
        for res_type in stack.defn.all_resource_types():
            self.enforce(stack.context, res_type, scope=scope, target=target, is_registered_policy=is_registered_policy)