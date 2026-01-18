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
def validate_stubout_neutron_create_security_group(self):
    self.m_csg.assert_called_once_with({'security_group': {'name': self.sg_name, 'description': 'HTTP and SSH access'}})
    self.validate_delete_security_group_rule()
    self.validate_create_security_group_rule_calls()