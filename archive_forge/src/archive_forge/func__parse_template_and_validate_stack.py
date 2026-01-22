import collections
import copy
import datetime
import functools
import itertools
import os
import pydoc
import signal
import socket
import sys
import eventlet
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_serialization import jsonutils
from oslo_service import service
from oslo_service import threadgroup
from oslo_utils import timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
import webob
from heat.common import context
from heat.common import environment_format as env_fmt
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import messaging as rpc_messaging
from heat.common import policy
from heat.common import service_utils
from heat.engine import api
from heat.engine import attributes
from heat.engine.cfn import template as cfntemplate
from heat.engine import clients
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine import parameter_groups
from heat.engine import properties
from heat.engine import resources
from heat.engine import service_software_config
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.engine import stk_defn
from heat.engine import support
from heat.engine import template as templatem
from heat.engine import template_files
from heat.engine import update
from heat.engine import worker
from heat.objects import event as event_object
from heat.objects import resource as resource_objects
from heat.objects import service as service_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_api as rpc_worker_api
def _parse_template_and_validate_stack(self, cnxt, stack_name, template, params, files, environment_files, files_container, args, owner_id=None, nested_depth=0, user_creds_id=None, stack_user_project_id=None, convergence=False, parent_resource_name=None, template_id=None):
    common_params = api.extract_args(args)
    if rpc_api.PARAM_ADOPT_STACK_DATA in common_params:
        if not cfg.CONF.enable_stack_adopt:
            raise exception.NotSupported(feature='Stack Adopt')
        new_params = {}
        if 'environment' in common_params[rpc_api.PARAM_ADOPT_STACK_DATA]:
            new_params = common_params[rpc_api.PARAM_ADOPT_STACK_DATA]['environment'].get(rpc_api.STACK_PARAMETERS, {}).copy()
        new_params.update(params.get(rpc_api.STACK_PARAMETERS, {}))
        params[rpc_api.STACK_PARAMETERS] = new_params
    if template_id is not None:
        tmpl = templatem.Template.load(cnxt, template_id)
    else:
        if files_container:
            files = template_files.get_files_from_container(cnxt, files_container, files)
        tmpl = templatem.Template(template, files=files)
        env_util.merge_environments(environment_files, files, params, tmpl.all_param_schemata(files))
        tmpl.env = environment.Environment(params)
    self._validate_new_stack(cnxt, stack_name, tmpl)
    stack = parser.Stack(cnxt, stack_name, tmpl, owner_id=owner_id, nested_depth=nested_depth, user_creds_id=user_creds_id, stack_user_project_id=stack_user_project_id, convergence=convergence, parent_resource=parent_resource_name, **common_params)
    self.resource_enforcer.enforce_stack(stack, is_registered_policy=True)
    self._validate_deferred_auth_context(cnxt, stack)
    is_root = stack.nested_depth == 0
    stack.validate()
    if is_root:
        tmpl.env.registry.log_resource_info(prefix=stack_name)
    return stack