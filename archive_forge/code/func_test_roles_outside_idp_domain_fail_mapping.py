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
def test_roles_outside_idp_domain_fail_mapping(self):
    d = unit.new_domain_ref()
    new_domain = PROVIDERS.resource_api.create_domain(d['id'], d)
    PROVIDERS.role_api.delete_role(self.member_role['id'])
    member_role_ref = unit.new_role_ref(name='member', domain_id=new_domain['id'])
    PROVIDERS.role_api.create_role(member_role_ref['id'], member_role_ref)
    self.assertRaises(exception.DomainSpecificRoleNotWithinIdPDomain, self._issue_unscoped_token)