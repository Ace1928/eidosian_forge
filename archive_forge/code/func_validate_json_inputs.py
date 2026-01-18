from unittest import mock
import yaml
from mistralclient.api import base as mistral_base
from mistralclient.api.v2 import executions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.mistral import workflow
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def validate_json_inputs(self, actual_input, expected_input):
    actual_json_input = jsonutils.loads(actual_input)
    expected_json_input = jsonutils.loads(expected_input)
    self.assertEqual(expected_json_input, actual_json_input)