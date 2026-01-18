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
def test_values_as_list_params_precedence(self):
    id = 1
    qp = 'query param!'
    qp2 = 'query param!!!!!'
    qp_name = 'query-param'
    uri_param = 'uri param!'
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.links = {}
    mock_response.json.return_value = {'resources': [{'id': id}]}
    mock_empty = mock.Mock()
    mock_empty.status_code = 200
    mock_empty.links = {}
    mock_empty.json.return_value = {'resources': []}
    self.session.get.side_effect = [mock_response, mock_empty]

    class Test(self.test_class):
        _query_mapping = resource.QueryParameters(query_param=qp_name)
        base_path = '/%(something)s/blah'
        something = resource.URI('something')
    results = list(Test.list(self.session, paginated=True, query_param=qp2, something=uri_param, **{qp_name: qp}))
    self.assertEqual(1, len(results))
    self.assertEqual(self.session.get.call_args_list[0][1]['params'], {qp_name: qp2})
    self.assertEqual(self.session.get.call_args_list[0][0][0], Test.base_path % {'something': uri_param})