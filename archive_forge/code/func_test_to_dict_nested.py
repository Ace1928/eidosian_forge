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
def test_to_dict_nested(self):

    class Test(resource.Resource):
        foo = resource.Header('foo')
        bar = resource.Body('bar')
        a_list = resource.Body('a_list')

    class Sub(resource.Resource):
        sub = resource.Body('foo')
    sub = Sub(id='ANOTHER_ID', foo='bar')
    res = Test(id='FAKE_ID', bar=sub, a_list=[sub])
    expected = {'id': 'FAKE_ID', 'name': None, 'location': None, 'foo': None, 'bar': {'id': 'ANOTHER_ID', 'name': None, 'sub': 'bar', 'location': None}, 'a_list': [{'id': 'ANOTHER_ID', 'name': None, 'sub': 'bar', 'location': None}]}
    self.assertEqual(expected, res.to_dict())
    a_munch = res.to_dict(_to_munch=True)
    self.assertEqual(a_munch.bar.id, 'ANOTHER_ID')
    self.assertEqual(a_munch.bar.sub, 'bar')
    self.assertEqual(a_munch.a_list[0].id, 'ANOTHER_ID')
    self.assertEqual(a_munch.a_list[0].sub, 'bar')