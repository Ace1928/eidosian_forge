import collections
import copy
from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_security_group_neutron(self):
    self.stubout_neutron_create_security_group()
    self.stubout_neutron_get_security_group()
    stack = self.create_stack(self.test_template_neutron)
    sg = stack['the_sg']
    self.assertResourceState(sg, 'aaaa')
    stack.delete()
    self.validate_stubout_neutron_create_security_group()
    self.m_ssg.assert_called_once_with('aaaa')
    self.m_dsg.assert_called_once_with('aaaa')