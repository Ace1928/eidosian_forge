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
def test_to_dict_ignore_none(self):

    class Test(resource.Resource):
        foo = resource.Header('foo')
        bar = resource.Body('bar')
    res = Test(id='FAKE_ID', bar='BAR')
    expected = {'id': 'FAKE_ID', 'bar': 'BAR'}
    self.assertEqual(expected, res.to_dict(ignore_none=True))