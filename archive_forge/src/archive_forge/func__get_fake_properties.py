import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _get_fake_properties(self, sg='one'):
    fake_groups_list = {'security_groups': [{'tenant_id': 'test_tenant_id', 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'name': 'security_group_1', 'security_group_rules': [], 'description': 'no protocol'}, {'tenant_id': 'test_tenant_id', 'id': '384ccd91-447c-4d83-832c-06974a7d3d05', 'name': 'security_group_2', 'security_group_rules': [], 'description': 'no protocol'}, {'tenant_id': 'test_tenant_id', 'id': 'e91a0007-06a6-4a4a-8edb-1d91315eb0ef', 'name': 'duplicate_group_name', 'security_group_rules': [], 'description': 'no protocol'}, {'tenant_id': 'test_tenant_id', 'id': '8be37f3c-176d-4826-aa17-77d1d9df7b2e', 'name': 'duplicate_group_name', 'security_group_rules': [], 'description': 'no protocol'}]}
    fixed_ip = {'subnet_id': 'fake_subnet_id'}
    props = {'admin_state_up': True, 'network_id': 'fake_network_id', 'fixed_ips': [fixed_ip], 'security_groups': ['0389f747-7785-4757-b7bb-2ab07e4b09c3']}
    if sg == 'zero':
        props['security_groups'] = []
    elif sg == 'one':
        props['security_groups'] = ['0389f747-7785-4757-b7bb-2ab07e4b09c3']
    elif sg == 'two':
        props['security_groups'] = ['0389f747-7785-4757-b7bb-2ab07e4b09c3', '384ccd91-447c-4d83-832c-06974a7d3d05']
    return (fake_groups_list, props)