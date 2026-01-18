import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_service_url(self):
    endpoint_filter = {'service_type': 'compute', 'interface': 'admin', 'service_name': 'nova'}
    self._do_service_url_test('http://nova/novapi/admin', endpoint_filter)