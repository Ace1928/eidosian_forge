from unittest import mock
from oslo_config import cfg
from requests import exceptions
import yaml
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import api
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.cfn import stack as stack_res
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_res_data_delete(self):
    stack = self.create_stack(self.tmpl)
    res = stack['res']
    stack.delete()
    self.assertEqual((stack.DELETE, stack.COMPLETE), stack.state)
    self.assertRaises(exception.NotFound, resource_data_object.ResourceData.get_val, res, 'test')