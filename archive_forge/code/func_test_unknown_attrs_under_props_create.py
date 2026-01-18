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
def test_unknown_attrs_under_props_create(self):

    class Test(resource.Resource):
        properties = resource.Body('properties')
        _store_unknown_attrs_as_properties = True
    sot = Test.new(**{'dummy': 'value'})
    self.assertDictEqual({'dummy': 'value'}, sot.properties)
    self.assertDictEqual({'dummy': 'value'}, sot.to_dict()['properties'])
    self.assertDictEqual({'dummy': 'value'}, sot['properties'])
    self.assertEqual('value', sot['properties']['dummy'])
    sot = Test.new(**{'dummy': 'value', 'properties': 'a,b,c'})
    self.assertDictEqual({'dummy': 'value', 'properties': 'a,b,c'}, sot.properties)
    self.assertDictEqual({'dummy': 'value', 'properties': 'a,b,c'}, sot.to_dict()['properties'])
    sot = Test.new(**{'properties': None})
    self.assertIsNone(sot.properties)
    self.assertIsNone(sot.to_dict()['properties'])