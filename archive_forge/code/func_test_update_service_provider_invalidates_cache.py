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
@unit.skip_if_cache_disabled('federation')
def test_update_service_provider_invalidates_cache(self):
    resp = self.get(self.base_url(), expected_status=http.client.OK)
    self.assertThat(resp.json_body['service_providers'], matchers.HasLength(1))
    service_provider_id = uuid.uuid4().hex
    url = self.base_url(suffix=service_provider_id)
    sp = core.new_service_provider_ref()
    self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
    resp = self.get(self.base_url(), expected_status=http.client.OK)
    self.assertThat(resp.json_body['service_providers'], matchers.HasLength(2))
    updated_description = uuid.uuid4().hex
    body = {'service_provider': {'description': updated_description}}
    self.patch(url, body=body, expected_status=http.client.OK)
    resp = self.get(self.base_url(), expected_status=http.client.OK)
    self.assertThat(resp.json_body['service_providers'], matchers.HasLength(2))
    for sp in resp.json_body['service_providers']:
        if sp['id'] == service_provider_id:
            self.assertEqual(sp['description'], updated_description)