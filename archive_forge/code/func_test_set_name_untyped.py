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
def test_set_name_untyped(self):
    name = 'name'
    expected_value = '123'

    class Parent:
        _example = {}
    instance = Parent()
    sot = TestComponent.ExampleComponent('name')
    sot.__set__(instance, expected_value)
    self.assertEqual(expected_value, instance._example[name])