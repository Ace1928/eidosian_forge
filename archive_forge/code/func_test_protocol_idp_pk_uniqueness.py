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
def test_protocol_idp_pk_uniqueness(self):
    """Test whether Keystone checks for unique idp/protocol values.

        Add same protocol twice, expect Keystone to reject a latter call and
        return HTTP 409 Conflict code.

        """
    url = self.base_url(suffix='%(idp_id)s/protocols/%(protocol_id)s')
    kwargs = {'expected_status': http.client.CREATED}
    resp, idp_id, proto = self._assign_protocol_to_idp(proto='saml2', url=url, **kwargs)
    kwargs = {'expected_status': http.client.CONFLICT}
    self._assign_protocol_to_idp(idp_id=idp_id, proto='saml2', validate=False, url=url, **kwargs)