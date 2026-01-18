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
def test_to_dict_with_unknown_attrs_in_body(self):

    class Test(resource.Resource):
        foo = resource.Body('foo')
        _allow_unknown_attrs_in_body = True
    res = Test(id='FAKE_ID', foo='FOO', bar='BAR')
    expected = {'id': 'FAKE_ID', 'name': None, 'location': None, 'foo': 'FOO', 'bar': 'BAR'}
    self.assertEqual(expected, res.to_dict())