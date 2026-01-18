import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_update_idp_clean_remote_ids(self):
    """Update IdP's remote_ids parameter with an empty list."""
    body = self.default_body.copy()
    body['remote_ids'] = [uuid.uuid4().hex]
    default_resp = self._create_default_idp(body=body)
    default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
    idp_id = default_idp.get('id')
    url = self.base_url(suffix=idp_id)
    self.assertIsNotNone(idp_id)
    body['remote_ids'] = []
    body = {'identity_provider': body}
    resp = self.patch(url, body=body)
    updated_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
    body = body['identity_provider']
    self.assertEqual(sorted(body['remote_ids']), sorted(updated_idp.get('remote_ids')))
    resp = self.get(url)
    returned_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
    self.assertEqual(sorted(body['remote_ids']), sorted(returned_idp.get('remote_ids')))