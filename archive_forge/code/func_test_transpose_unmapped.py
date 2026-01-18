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
def test_transpose_unmapped(self):

    def _type(value, rtype):
        self.assertIs(rtype, mock.sentinel.resource_type)
        return value * 10
    location = 'location'
    mapping = {'first_name': 'first-name', 'pet_name': {'name': 'pet'}, 'answer': {'name': 'answer', 'type': int}, 'complex': {'type': _type}}
    sot = resource.QueryParameters(location, **mapping)
    result = sot._transpose({'location': 'Brooklyn', 'first_name': 'Brian', 'pet_name': 'Meow', 'answer': '42', 'last_name': 'Curtin', 'complex': 1}, mock.sentinel.resource_type)
    self.assertEqual({'location': 'Brooklyn', 'first-name': 'Brian', 'pet': 'Meow', 'answer': 42, 'complex': 10}, result)