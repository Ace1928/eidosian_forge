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
def _test_bulk_create(self, cls, http_method, microversion=None, base_path=None, **params):
    req1 = mock.Mock()
    req2 = mock.Mock()
    req1.body = {'name': 'resource1'}
    req2.body = {'name': 'resource2'}
    req1.url = 'uri'
    req2.url = 'uri'
    req1.headers = 'headers'
    req2.headers = 'headers'
    request_body = {'tests': [{'name': 'resource1', 'id': 'id1'}, {'name': 'resource2', 'id': 'id2'}]}
    cls._prepare_request = mock.Mock(side_effect=[req1, req2])
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.links = {}
    mock_response.json.return_value = request_body
    http_method.return_value = mock_response
    res = list(cls.bulk_create(self.session, [{'name': 'resource1'}, {'name': 'resource2'}], base_path=base_path, **params))
    self.assertEqual(len(res), 2)
    self.assertEqual(res[0].id, 'id1')
    self.assertEqual(res[1].id, 'id2')
    http_method.assert_called_once_with(self.request.url, json={'tests': [req1.body, req2.body]}, headers=self.request.headers, microversion=microversion, params=params)