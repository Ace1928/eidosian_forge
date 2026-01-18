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
def test_status_fails(self):
    failure = 'crying'
    statuses = ['success', 'other', failure]
    res = self._fake_resource(statuses)
    self.assertRaises(exceptions.ResourceFailure, resource.wait_for_status, mock.Mock(), res, 'loling', [failure], interval=1, wait=5)