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
def test_status_match_different_attribute(self):
    status = 'loling'
    statuses = ['first', 'other', 'another', 'another', status]
    res = self._fake_resource(statuses, attribute='mood')
    result = resource.wait_for_status(mock.Mock(), res, status, None, interval=1, wait=5, attribute='mood')
    self.assertEqual(result, res)