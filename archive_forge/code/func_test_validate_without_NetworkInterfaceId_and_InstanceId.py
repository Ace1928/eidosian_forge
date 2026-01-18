import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_validate_without_NetworkInterfaceId_and_InstanceId(self):
    server = self.fc.servers.list()[0]
    self.patchobject(self.fc.servers, 'get', return_value=server)
    template, stack = self._setup_test_stack_validate(stack_name='validate_EIP_InstanceId')
    properties = template.t['Resources']['IPAssoc']['Properties']
    properties.pop('InstanceId')
    properties.pop('EIP')
    allocation_id = '1fafbe59-2332-4f5f-bfa4-517b4d6c1b65'
    properties['AllocationId'] = allocation_id
    resource_defns = template.resource_definitions(stack)
    rsrc = eip.ElasticIpAssociation('validate_eip_ass', resource_defns['IPAssoc'], stack)
    exc = self.assertRaises(exception.PropertyUnspecifiedError, rsrc.validate)
    self.assertIn('At least one of the following properties must be specified: InstanceId, NetworkInterfaceId', str(exc))