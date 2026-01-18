import json
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
def update_with_template(self, child_template, user_params=None, timeout_mins=None):
    """Update the nested stack with the new template."""
    if self.id is None:
        self.store()
    if self.stack.action == self.stack.ROLLBACK:
        if self._try_rollback():
            LOG.info('Triggered nested stack %s rollback', self.physical_resource_name())
            return {'target_action': self.stack.ROLLBACK}
    if self.resource_id is None:

        def _check_for_completion():
            while not self.check_create_complete():
                yield
        empty_temp = template_format.parse("heat_template_version: '2013-05-23'")
        self.create_with_template(empty_temp, {})
        checker = scheduler.TaskRunner(_check_for_completion)
        checker(timeout=self.stack.timeout_secs())
    if timeout_mins is None:
        timeout_mins = self.stack.timeout_mins
    try:
        status_data = stack_object.Stack.get_status(self.context, self.resource_id)
    except exception.NotFound:
        raise resource.UpdateReplace(self)
    action, status, status_reason, updated_time = status_data
    kwargs = self._stack_kwargs(user_params, child_template)
    kwargs.update({'stack_identity': dict(self.nested_identifier()), 'args': {rpc_api.PARAM_TIMEOUT: timeout_mins, rpc_api.PARAM_CONVERGE: self.converge}})
    with self.translate_remote_exceptions:
        try:
            self.rpc_client()._update_stack(self.context, **kwargs)
        except exception.HeatException:
            with excutils.save_and_reraise_exception():
                raw_template.RawTemplate.delete(self.context, kwargs['template_id'])