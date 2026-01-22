import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def _compare_servers(self, exp, real):
    self.assertDictEqual(server.Server(**exp).to_dict(computed=False), real.to_dict(computed=False))