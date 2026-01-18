import collections
import copy
import os.path
from oslo_serialization import jsonutils
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import environments
from mistralclient.tests.unit.v2 import base
from mistralclient import utils
def test_update_without_name(self):
    data = copy.deepcopy(ENVIRONMENT)
    data.pop('name')
    e = self.assertRaises(api_base.APIException, self.environments.update, **data)
    self.assertEqual(400, e.error_code)