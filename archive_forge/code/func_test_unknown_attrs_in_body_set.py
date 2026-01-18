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
def test_unknown_attrs_in_body_set(self):

    class Test(resource.Resource):
        known_param = resource.Body('known_param')
        _allow_unknown_attrs_in_body = True
    sot = Test.new(**{'known_param': 'v1'})
    sot['unknown_param'] = 'v2'
    self.assertEqual('v1', sot.known_param)
    self.assertEqual('v2', sot.unknown_param)