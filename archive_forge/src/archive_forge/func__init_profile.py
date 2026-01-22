from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import profile as sp
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _init_profile(self, template):
    self.stack = utils.parse_stack(template)
    profile = self.stack['senlin-profile']
    return profile