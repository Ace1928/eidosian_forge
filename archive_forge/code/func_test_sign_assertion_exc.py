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
@mock.patch('oslo_utils.fileutils.write_to_tempfile')
def test_sign_assertion_exc(self, write_to_tempfile_mock):
    sample_returncode = 1
    sample_output = self.getUniqueString()
    write_to_tempfile_mock.return_value = 'tmp_path'

    def side_effect(*args, **kwargs):
        if args[0] == ['/usr/bin/which', CONF.saml.xmlsec1_binary]:
            return '/usr/bin/xmlsec1\n'
        else:
            raise subprocess.CalledProcessError(returncode=sample_returncode, cmd=CONF.saml.xmlsec1_binary, output=sample_output)
    with mock.patch.object(subprocess, 'check_output', side_effect=side_effect):
        logger_fixture = self.useFixture(fixtures.LoggerFixture())
        self.assertRaises(exception.SAMLSigningError, keystone_idp._sign_assertion, self.signed_assertion)
        expected_log = "Error when signing assertion, reason: Command '%s' returned non-zero exit status %s\\.? %s\\n" % (CONF.saml.xmlsec1_binary, sample_returncode, sample_output)
        self.assertRegex(logger_fixture.output, re.compile('%s' % expected_log))