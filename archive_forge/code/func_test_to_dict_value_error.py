import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_to_dict_value_error(self):

    class Test(resource.Resource):
        foo = resource.Header('foo')
        bar = resource.Body('bar')
    res = Test(id='FAKE_ID')
    err = self.assertRaises(ValueError, res.to_dict, body=False, headers=False, computed=False)
    self.assertEqual('At least one of `body`, `headers` or `computed` must be True', str(err))