from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from urllib import parse as urlparse
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine.resources.openstack.heat import deployed_server
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _setup_test_stack(self, stack_name, test_templ=ds_tmpl):
    t = template_format.parse(test_templ)
    tmpl = template.Template(t, env=environment.Environment())
    stack = parser.Stack(utils.dummy_context(region_name='RegionOne'), stack_name, tmpl, stack_id=uuidutils.generate_uuid(), stack_user_project_id='8888')
    return (tmpl, stack)