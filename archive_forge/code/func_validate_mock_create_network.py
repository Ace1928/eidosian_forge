from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def validate_mock_create_network(self):
    self.mockclient.show_network.assert_called_with('aaaa')
    self.mockclient.create_network.assert_called_once_with({'network': {'name': self.vpc_name}})
    self.mockclient.create_router.assert_called_once()