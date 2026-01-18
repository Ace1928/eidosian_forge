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
def test_setitem_unchanged(self):
    key = 'key'
    value = 'value'
    attrs = {key: value}
    sot = resource._ComponentManager(attributes=attrs, synchronized=True)
    sot.__setitem__(key, value)
    self.assertEqual(value, sot.attributes[key])
    self.assertNotIn(key, sot.dirty)