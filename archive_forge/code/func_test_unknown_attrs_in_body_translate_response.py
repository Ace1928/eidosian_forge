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
def test_unknown_attrs_in_body_translate_response(self):

    class Test(resource.Resource):
        known_param = resource.Body('known_param')
        _allow_unknown_attrs_in_body = True
    body = {'known_param': 'v1', 'unknown_param': 'v2'}
    response = FakeResponse(body)
    sot = Test()
    sot._translate_response(response, has_body=True)
    self.assertEqual('v1', sot.known_param)
    self.assertEqual('v2', sot.unknown_param)