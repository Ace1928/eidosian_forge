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
def test_serialize_list(self):
    original = {'test_key': [1, 2, 3]}
    expected = {'test_key': '[1, 2, 3]'}
    self.assertEqual(expected, self.nova_plugin.meta_serialize(original))