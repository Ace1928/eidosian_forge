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
def test_update_protocols_attribute(self):
    """Update protocol's attribute."""
    resp, idp_id, proto = self._assign_protocol_to_idp(expected_status=http.client.CREATED)
    new_mapping_id = uuid.uuid4().hex
    self._create_mapping(mapping_id=new_mapping_id)
    url = '%s/protocols/%s' % (idp_id, proto)
    url = self.base_url(suffix=url)
    body = {'mapping_id': new_mapping_id}
    resp = self.patch(url, body={'protocol': body})
    self.assertValidResponse(resp, 'protocol', dummy_validator, keys_to_check=['id', 'mapping_id'], ref={'id': proto, 'mapping_id': new_mapping_id})