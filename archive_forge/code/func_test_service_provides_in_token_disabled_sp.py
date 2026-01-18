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
def test_service_provides_in_token_disabled_sp(self):
    """Test behaviour with disabled service providers.

        Disabled service providers should not be listed in the service
        catalog.

        """
    sp_ref = {'enabled': False}
    PROVIDERS.federation_api.update_sp(self.SP1, sp_ref)
    model = token_model.TokenModel()
    model.user_id = self.user_id
    model.methods = ['password']
    token = render_token.render_token_response_from_model(model)
    ref = {}
    for r in (self.sp_beta, self.sp_gamma):
        ref.update(r)
    self._validate_service_providers(token, ref)