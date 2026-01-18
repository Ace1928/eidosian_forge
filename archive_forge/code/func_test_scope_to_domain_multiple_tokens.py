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
def test_scope_to_domain_multiple_tokens(self):
    """Issue multiple tokens scoping to different domains.

        The new tokens should be scoped to:

        * domainA
        * domainB
        * domainC

        """
    bodies = (self.TOKEN_SCOPE_DOMAIN_A_FROM_ADMIN, self.TOKEN_SCOPE_DOMAIN_B_FROM_ADMIN, self.TOKEN_SCOPE_DOMAIN_C_FROM_ADMIN)
    domain_ids = (self.domainA['id'], self.domainB['id'], self.domainC['id'])
    for body, domain_id_ref in zip(bodies, domain_ids):
        r = self.v3_create_token(body)
        token_resp = r.result['token']
        self._check_domain_scoped_token_attributes(token_resp, domain_id_ref)