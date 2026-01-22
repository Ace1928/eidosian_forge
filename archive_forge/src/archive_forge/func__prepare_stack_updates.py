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
def _prepare_stack_updates(self, cnxt, current_stack, template, params, environment_files, files, files_container, args, template_id=None):
    """Return the current and updated stack for a given transition.

        Changes *will not* be persisted, this is a helper method for
        update_stack and preview_update_stack.

        :param cnxt: RPC context.
        :param stack: A stack to be updated.
        :param template: Template of stack you want to update to.
        :param params: Stack Input Params
        :param files: Files referenced from the template
        :param args: Request parameters/args passed from API
        :param template_id: the ID of a pre-stored template in the DB
        """
    if args.get(rpc_api.PARAM_EXISTING, False):
        assert template_id is None, 'Cannot specify template_id with PARAM_EXISTING'
        if template is not None:
            new_template = template
        elif current_stack.convergence or current_stack.status == current_stack.COMPLETE:
            new_template = current_stack.t.t
        elif current_stack.prev_raw_template_id is not None:
            prev_t = templatem.Template.load(cnxt, current_stack.prev_raw_template_id)
            new_template = prev_t.t
        else:
            LOG.error('PATCH update to FAILED stack only possible if convergence enabled or previous template stored')
            msg = _('PATCH update to non-COMPLETE stack')
            raise exception.NotSupported(feature=msg)
        new_files = current_stack.t.files
        if files_container:
            files = template_files.get_files_from_container(cnxt, files_container, files)
        new_files.update(files or {})
        tmpl = templatem.Template(new_template, files=new_files)
        env_util.merge_environments(environment_files, new_files, params, tmpl.all_param_schemata(files))
        existing_env = current_stack.env.env_as_dict()
        existing_params = existing_env[env_fmt.PARAMETERS]
        clear_params = set(args.get(rpc_api.PARAM_CLEAR_PARAMETERS, []))
        retained = dict(((k, v) for k, v in existing_params.items() if k not in clear_params))
        existing_env[env_fmt.PARAMETERS] = retained
        new_env = environment.Environment(existing_env)
        new_env.load(params)
        for key in list(new_env.params.keys()):
            if key not in tmpl.param_schemata():
                new_env.params.pop(key)
        tmpl.env = new_env
    elif template_id is not None:
        tmpl = templatem.Template.load(cnxt, template_id)
    else:
        if files_container:
            files = template_files.get_files_from_container(cnxt, files_container, files)
        tmpl = templatem.Template(template, files=files)
        env_util.merge_environments(environment_files, files, params, tmpl.all_param_schemata(files))
        tmpl.env = environment.Environment(params)
    max_resources = cfg.CONF.max_resources_per_stack
    if max_resources != -1 and len(tmpl[tmpl.RESOURCES]) > max_resources:
        raise exception.RequestLimitExceeded(message=exception.StackResourceLimitExceeded.msg_fmt)
    stack_name = current_stack.name
    current_kwargs = current_stack.get_kwargs_for_cloning()
    common_params = api.extract_args(args)
    common_params.setdefault(rpc_api.PARAM_TIMEOUT, current_stack.timeout_mins)
    common_params.setdefault(rpc_api.PARAM_DISABLE_ROLLBACK, current_stack.disable_rollback)
    common_params.setdefault(rpc_api.PARAM_CONVERGE, current_stack.converge)
    if args.get(rpc_api.PARAM_EXISTING, False):
        if rpc_api.STACK_TAGS not in common_params:
            common_params[rpc_api.STACK_TAGS] = current_stack.tags
    current_kwargs.update(common_params)
    updated_stack = parser.Stack(cnxt, stack_name, tmpl, **current_kwargs)
    invalid_params = current_stack.parameters.immutable_params_modified(updated_stack.parameters, tmpl.env.params)
    if invalid_params:
        raise exception.ImmutableParameterModified(*invalid_params)
    self.resource_enforcer.enforce_stack(updated_stack, is_registered_policy=True)
    updated_stack.parameters.set_stack_id(current_stack.identifier())
    self._validate_deferred_auth_context(cnxt, updated_stack)
    updated_stack.validate()
    return (tmpl, current_stack, updated_stack)