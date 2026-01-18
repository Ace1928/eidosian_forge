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
def test_unknown_attrs_not_stored(self):

    class Test(resource.Resource):
        properties = resource.Body('properties')
    sot = Test.new(**{'dummy': 'value'})
    self.assertIsNone(sot.properties)