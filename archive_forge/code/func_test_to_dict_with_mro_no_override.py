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
def test_to_dict_with_mro_no_override(self):

    class Parent(resource.Resource):
        header = resource.Header('HEADER')
        body = resource.Body('BODY')

    class Child(Parent):
        header = resource.Header('ANOTHER_HEADER')
        body = resource.Body('ANOTHER_BODY')
    res = Child(id='FAKE_ID', body='BODY_VALUE', header='HEADER_VALUE')
    expected = {'body': 'BODY_VALUE', 'header': 'HEADER_VALUE', 'id': 'FAKE_ID', 'location': None, 'name': None}
    self.assertEqual(expected, res.to_dict())