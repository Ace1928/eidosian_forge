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
def test_unknown_attrs_prepare_request_unpacked(self):

    class Test(resource.Resource):
        properties = resource.Body('properties')
        _store_unknown_attrs_as_properties = True
    sot = Test.new(**{'dummy': 'value', 'properties': 'a,b,c'})
    request_body = sot._prepare_request(requires_id=False).body
    self.assertEqual('value', request_body['dummy'])
    self.assertEqual('a,b,c', request_body['properties'])
    sot = Test.new(**{'properties': {'properties': 'a,b,c', 'dummy': 'value'}})
    request_body = sot._prepare_request(requires_id=False).body
    self.assertEqual('value', request_body['dummy'])
    self.assertEqual('a,b,c', request_body['properties'])