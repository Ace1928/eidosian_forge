import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_setting_global_id_on_request(self):
    global_id_adpt = 'req-%s' % uuid.uuid4()
    global_id_req = 'req-%s' % uuid.uuid4()
    response = uuid.uuid4().hex
    self.stub_url('GET', text=response)

    def mk_adpt(**kwargs):
        return adapter.Adapter(client_session.Session(), auth=CalledAuthPlugin(), service_type=self.SERVICE_TYPE, service_name=self.SERVICE_NAME, interface=self.INTERFACE, region_name=self.REGION_NAME, user_agent=self.USER_AGENT, version=self.VERSION, allow=self.ALLOW, **kwargs)
    adpt = mk_adpt()
    resp = adpt.get('/')
    self.assertEqual(resp.text, response)
    self._verify_endpoint_called(adpt)
    self.assertEqual(self.ALLOW, adpt.auth.endpoint_arguments['allow'])
    self.assertTrue(adpt.auth.get_token_called)
    self.assertRequestHeaderEqual('X-OpenStack-Request-ID', None)
    adpt.get('/', global_request_id=global_id_req)
    self.assertRequestHeaderEqual('X-OpenStack-Request-ID', global_id_req)
    adpt = mk_adpt(global_request_id=global_id_adpt)
    adpt.get('/')
    self.assertRequestHeaderEqual('X-OpenStack-Request-ID', global_id_adpt)
    adpt.get('/', global_request_id=global_id_req)
    self.assertRequestHeaderEqual('X-OpenStack-Request-ID', global_id_req)