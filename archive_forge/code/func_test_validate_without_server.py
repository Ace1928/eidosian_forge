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
def test_validate_without_server(self):
    stack = utils.parse_stack(self.template_no_server)
    snip = stack.t.resource_definitions(stack)['deployment_mysql']
    deployment = sd.SoftwareDeployment('deployment_mysql', snip, stack)
    err = self.assertRaises(exc.StackValidationFailed, deployment.validate)
    self.assertEqual('Property error: Resources.deployment_mysql.Properties: Property server not assigned', str(err))