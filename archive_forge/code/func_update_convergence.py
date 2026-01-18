import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
def update_convergence(self, template_id, new_requires, engine_id, timeout, new_stack, progress_callback=None):
    """Update the resource synchronously.

        Persist the resource's current_template_id to template_id and
        resource's requires to list of the required resource ids from the given
        resource_data and existing resource's requires, then updates the
        resource by invoking the scheduler TaskRunner.
        """
    self._calling_engine_id = engine_id
    registry = new_stack.env.registry
    new_res_def = new_stack.defn.resource_definition(self.name)
    new_res_type = registry.get_class_to_instantiate(new_res_def.resource_type, resource_name=self.name)
    if type(self) is not new_res_type:
        restrictions = registry.get_rsrc_restricted_actions(self.name)
        self._check_for_convergence_replace(restrictions)
    action_rollback = self.stack.action == self.stack.ROLLBACK
    status_in_progress = self.stack.status == self.stack.IN_PROGRESS
    if action_rollback and status_in_progress and self.replaced_by:
        try:
            self.restore_prev_rsrc(convergence=True)
        except Exception as e:
            failure = exception.ResourceFailure(e, self, self.action)
            self.state_set(self.UPDATE, self.FAILED, str(failure))
            raise failure
    self.replaced_by = None
    runner = scheduler.TaskRunner(self.update, new_res_def, new_template_id=template_id, new_requires=new_requires)
    runner(timeout=timeout, progress_callback=progress_callback)