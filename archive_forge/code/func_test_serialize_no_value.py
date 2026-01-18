import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_serialize_no_value(self):
    """Prove that the user can only pass in a dict to nova metadata."""
    excp = self.assertRaises(exception.StackValidationFailed, self.nova_plugin.meta_serialize, 'foo')
    self.assertIn('metadata needs to be a Map', str(excp))