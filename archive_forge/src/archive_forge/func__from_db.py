import collections
import contextlib
import copy
import eventlet
import functools
import re
import warnings
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import timeutils as oslo_timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
from heat.common import context as common_context
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import lifecycle_plugin_utils
from heat.engine import api
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import event
from heat.engine.notification import stack as notification
from heat.engine import parameter_groups as param_groups
from heat.engine import parent_rsrc
from heat.engine import resource
from heat.engine import resources
from heat.engine import scheduler
from heat.engine import status
from heat.engine import stk_defn
from heat.engine import sync_point
from heat.engine import template as tmpl
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_client as rpc_worker_client
@classmethod
def _from_db(cls, context, stack, use_stored_context=False, cache_data=None, load_template=True, refresh_cred=False):
    if load_template:
        template = tmpl.Template.load(context, stack.raw_template_id, stack.raw_template)
    else:
        template = None
    return cls(context, stack.name, template, stack_id=stack.id, action=stack.action, status=stack.status, status_reason=stack.status_reason, timeout_mins=stack.timeout, disable_rollback=stack.disable_rollback, parent_resource=stack.parent_resource_name, owner_id=stack.owner_id, stack_user_project_id=stack.stack_user_project_id, created_time=stack.created_at, updated_time=stack.updated_at, user_creds_id=stack.user_creds_id, tenant_id=stack.tenant, use_stored_context=use_stored_context, username=stack.username, convergence=stack.convergence, current_traversal=stack.current_traversal, prev_raw_template_id=stack.prev_raw_template_id, current_deps=stack.current_deps, cache_data=cache_data, nested_depth=stack.nested_depth, deleted_time=stack.deleted_at, refresh_cred=refresh_cred)