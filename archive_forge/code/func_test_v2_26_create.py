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
def test_v2_26_create(self):
    ctxt = utils.dummy_context()
    self.patchobject(nc, 'Client', return_value=mock.Mock())
    plugin = ctxt.clients.client_plugin('nova')
    plugin.max_microversion = '2.53'
    plugin.client(version='2.26')