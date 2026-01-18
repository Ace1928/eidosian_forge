from unittest import mock
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine.resources.openstack.swift import container as swift_c
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_refid(self):
    stack = utils.parse_stack(self.t)
    rsrc = stack['SwiftContainer']
    rsrc.resource_id = 'xyz'
    self.assertEqual('xyz', rsrc.FnGetRefId())