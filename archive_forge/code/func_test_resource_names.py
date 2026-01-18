import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_names(self):
    stack = utils.parse_stack(self.template)
    snip = stack.t.resource_definitions(stack)['deploy_mysql']
    resg = sd.SoftwareDeploymentGroup('test', snip, stack)
    self.assertEqual(set(('server1', 'server2')), set(resg._resource_names()))
    resg.properties = {'servers': {'s1': 'u1', 's2': 'u2', 's3': 'u3'}}
    self.assertEqual(set(('s1', 's2', 's3')), set(resg._resource_names()))