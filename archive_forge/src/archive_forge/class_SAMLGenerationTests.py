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
class SAMLGenerationTests(test_v3.RestfulTestCase):
    SP_AUTH_URL = 'http://beta.com:5000/v3/OS-FEDERATION/identity_providers/BETA/protocols/saml2/auth'
    ASSERTION_FILE = 'signed_saml2_assertion.xml'
    ISSUER = 'https://acme.com/FIM/sps/openstack/saml20'
    RECIPIENT = 'http://beta.com/Shibboleth.sso/SAML2/POST'
    SUBJECT = 'test_user'
    SUBJECT_DOMAIN = 'user_domain'
    ROLES = ['admin', 'member']
    PROJECT = 'development'
    PROJECT_DOMAIN = 'project_domain'
    GROUPS = ['JSON:{"name":"group1","domain":{"name":"Default"}}', 'JSON:{"name":"group2","domain":{"name":"Default"}}']
    SAML_GENERATION_ROUTE = '/auth/OS-FEDERATION/saml2'
    ECP_GENERATION_ROUTE = '/auth/OS-FEDERATION/saml2/ecp'
    ASSERTION_VERSION = '2.0'
    SERVICE_PROVDIER_ID = 'ACME'

    def setUp(self):
        super(SAMLGenerationTests, self).setUp()
        self.signed_assertion = saml2.create_class_from_xml_string(saml.Assertion, _load_xml(self.ASSERTION_FILE))
        self.sp = core.new_service_provider_ref(auth_url=self.SP_AUTH_URL, sp_url=self.RECIPIENT)
        url = '/OS-FEDERATION/service_providers/' + self.SERVICE_PROVDIER_ID
        self.put(url, body={'service_provider': self.sp}, expected_status=http.client.CREATED)

    def test_samlize_token_values(self):
        """Test the SAML generator produces a SAML object.

        Test the SAML generator directly by passing known arguments, the result
        should be a SAML object that consistently includes attributes based on
        the known arguments that were passed in.

        """
        with mock.patch.object(keystone_idp, '_sign_assertion', return_value=self.signed_assertion):
            generator = keystone_idp.SAMLGenerator()
            response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
        assertion = response.assertion
        self.assertIsNotNone(assertion)
        self.assertIsInstance(assertion, saml.Assertion)
        issuer = response.issuer
        self.assertEqual(self.RECIPIENT, response.destination)
        self.assertEqual(self.ISSUER, issuer.text)
        user_attribute = assertion.attribute_statement[0].attribute[0]
        self.assertEqual(self.SUBJECT, user_attribute.attribute_value[0].text)
        user_domain_attribute = assertion.attribute_statement[0].attribute[1]
        self.assertEqual(self.SUBJECT_DOMAIN, user_domain_attribute.attribute_value[0].text)
        role_attribute = assertion.attribute_statement[0].attribute[2]
        for attribute_value in role_attribute.attribute_value:
            self.assertIn(attribute_value.text, self.ROLES)
        project_attribute = assertion.attribute_statement[0].attribute[3]
        self.assertEqual(self.PROJECT, project_attribute.attribute_value[0].text)
        project_domain_attribute = assertion.attribute_statement[0].attribute[4]
        self.assertEqual(self.PROJECT_DOMAIN, project_domain_attribute.attribute_value[0].text)
        group_attribute = assertion.attribute_statement[0].attribute[5]
        for attribute_value in group_attribute.attribute_value:
            self.assertIn(attribute_value.text, self.GROUPS)

    def test_comma_in_certfile_path(self):
        self.config_fixture.config(group='saml', certfile=CONF.saml.certfile + ',')
        generator = keystone_idp.SAMLGenerator()
        self.assertRaises(exception.UnexpectedError, generator.samlize_token, self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)

    def test_comma_in_keyfile_path(self):
        self.config_fixture.config(group='saml', keyfile=CONF.saml.keyfile + ',')
        generator = keystone_idp.SAMLGenerator()
        self.assertRaises(exception.UnexpectedError, generator.samlize_token, self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)

    def test_verify_assertion_object(self):
        """Test that the Assertion object is built properly.

        The Assertion doesn't need to be signed in this test, so
        _sign_assertion method is patched and doesn't alter the assertion.

        """
        with mock.patch.object(keystone_idp, '_sign_assertion', side_effect=lambda x: x):
            generator = keystone_idp.SAMLGenerator()
            response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
        assertion = response.assertion
        self.assertEqual(self.ASSERTION_VERSION, assertion.version)

    def test_valid_saml_xml(self):
        """Test the generated SAML object can become valid XML.

        Test the generator directly by passing known arguments, the result
        should be a SAML object that consistently includes attributes based on
        the known arguments that were passed in.

        """
        with mock.patch.object(keystone_idp, '_sign_assertion', return_value=self.signed_assertion):
            generator = keystone_idp.SAMLGenerator()
            response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
        saml_str = response.to_string()
        response = etree.fromstring(saml_str)
        issuer = response[0]
        assertion = response[2]
        self.assertEqual(self.RECIPIENT, response.get('Destination'))
        self.assertEqual(self.ISSUER, issuer.text)
        user_attribute = assertion[4][0]
        self.assertEqual(self.SUBJECT, user_attribute[0].text)
        user_domain_attribute = assertion[4][1]
        self.assertEqual(self.SUBJECT_DOMAIN, user_domain_attribute[0].text)
        role_attribute = assertion[4][2]
        for attribute_value in role_attribute:
            self.assertIn(attribute_value.text, self.ROLES)
        project_attribute = assertion[4][3]
        self.assertEqual(self.PROJECT, project_attribute[0].text)
        project_domain_attribute = assertion[4][4]
        self.assertEqual(self.PROJECT_DOMAIN, project_domain_attribute[0].text)
        group_attribute = assertion[4][5]
        for attribute_value in group_attribute:
            self.assertIn(attribute_value.text, self.GROUPS)

    def test_assertion_using_explicit_namespace_prefixes(self):

        def mocked_subprocess_check_output(*popenargs, **kwargs):
            if popenargs[0] != ['/usr/bin/which', CONF.saml.xmlsec1_binary]:
                filename = popenargs[0][-1]
                with open(filename, 'r') as f:
                    assertion_content = f.read()
                return assertion_content
        with mock.patch.object(subprocess, 'check_output', side_effect=mocked_subprocess_check_output):
            generator = keystone_idp.SAMLGenerator()
            response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
            assertion_xml = response.assertion.to_string()
            self.assertIn(b'<saml:Assertion', assertion_xml)
            self.assertIn(('xmlns:saml="' + saml2.NAMESPACE + '"').encode('utf-8'), assertion_xml)
            self.assertIn(('xmlns:xmldsig="' + xmldsig.NAMESPACE).encode('utf-8'), assertion_xml)

    def test_saml_signing(self):
        """Test that the SAML generator produces a SAML object.

        Test the SAML generator directly by passing known arguments, the result
        should be a SAML object that consistently includes attributes based on
        the known arguments that were passed in.

        """
        if not _is_xmlsec1_installed():
            self.skipTest('xmlsec1 is not installed')
        generator = keystone_idp.SAMLGenerator()
        response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
        signature = response.assertion.signature
        self.assertIsNotNone(signature)
        self.assertIsInstance(signature, xmldsig.Signature)
        idp_public_key = sigver.read_cert_from_file(CONF.saml.certfile, 'pem')
        cert_text = signature.key_info.x509_data[0].x509_certificate.text
        cert_text = cert_text.replace(os.linesep, '')
        self.assertEqual(idp_public_key, cert_text)

    def _create_generate_saml_request(self, token_id, sp_id):
        return {'auth': {'identity': {'methods': ['token'], 'token': {'id': token_id}}, 'scope': {'service_provider': {'id': sp_id}}}}

    def _fetch_valid_token(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        resp = self.v3_create_token(auth_data)
        token_id = resp.headers.get('X-Subject-Token')
        return token_id

    def _fetch_domain_scoped_token(self):
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], user_domain_id=self.domain['id'])
        resp = self.v3_create_token(auth_data)
        token_id = resp.headers.get('X-Subject-Token')
        return token_id

    def test_not_project_scoped_token(self):
        """Ensure SAML generation fails when passing domain-scoped tokens.

        The server should return a 403 Forbidden Action.

        """
        self.config_fixture.config(group='saml', idp_entity_id=self.ISSUER)
        token_id = self._fetch_domain_scoped_token()
        body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
        with mock.patch.object(keystone_idp, '_sign_assertion', return_value=self.signed_assertion):
            self.post(self.SAML_GENERATION_ROUTE, body=body, expected_status=http.client.FORBIDDEN)

    def test_generate_saml_route(self):
        """Test that the SAML generation endpoint produces XML.

        The SAML endpoint /v3/auth/OS-FEDERATION/saml2 should take as input,
        a scoped token ID, and a Service Provider ID.
        The controller should fetch details about the user from the token,
        and details about the service provider from its ID.
        This should be enough information to invoke the SAML generator and
        provide a valid SAML (XML) document back.

        """
        self.config_fixture.config(group='saml', idp_entity_id=self.ISSUER)
        token_id = self._fetch_valid_token()
        body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
        with mock.patch.object(keystone_idp, '_sign_assertion', return_value=self.signed_assertion):
            http_response = self.post(self.SAML_GENERATION_ROUTE, body=body, response_content_type='text/xml', expected_status=http.client.OK)
        response = etree.fromstring(http_response.result)
        issuer = response[0]
        assertion = response[2]
        self.assertEqual(self.RECIPIENT, response.get('Destination'))
        self.assertEqual(self.ISSUER, issuer.text)
        user_attribute = assertion[4][0]
        self.assertIsInstance(user_attribute[0].text, str)
        user_domain_attribute = assertion[4][1]
        self.assertIsInstance(user_domain_attribute[0].text, str)
        role_attribute = assertion[4][2]
        self.assertIsInstance(role_attribute[0].text, str)
        project_attribute = assertion[4][3]
        self.assertIsInstance(project_attribute[0].text, str)
        project_domain_attribute = assertion[4][4]
        self.assertIsInstance(project_domain_attribute[0].text, str)
        group_attribute = assertion[4][5]
        self.assertIsInstance(group_attribute[0].text, str)

    def test_invalid_scope_body(self):
        """Test that missing the scope in request body raises an exception.

        Raises exception.SchemaValidationError() - error 400 Bad Request

        """
        token_id = uuid.uuid4().hex
        body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
        del body['auth']['scope']
        self.post(self.SAML_GENERATION_ROUTE, body=body, expected_status=http.client.BAD_REQUEST)

    def test_invalid_token_body(self):
        """Test that missing the token in request body raises an exception.

        Raises exception.SchemaValidationError() - error 400 Bad Request

        """
        token_id = uuid.uuid4().hex
        body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
        del body['auth']['identity']['token']
        self.post(self.SAML_GENERATION_ROUTE, body=body, expected_status=http.client.BAD_REQUEST)

    def test_sp_not_found(self):
        """Test SAML generation with an invalid service provider ID.

        Raises exception.ServiceProviderNotFound() - error Not Found 404

        """
        sp_id = uuid.uuid4().hex
        token_id = self._fetch_valid_token()
        body = self._create_generate_saml_request(token_id, sp_id)
        self.post(self.SAML_GENERATION_ROUTE, body=body, expected_status=http.client.NOT_FOUND)

    def test_sp_disabled(self):
        """Try generating assertion for disabled Service Provider."""
        sp_ref = {'enabled': False}
        PROVIDERS.federation_api.update_sp(self.SERVICE_PROVDIER_ID, sp_ref)
        token_id = self._fetch_valid_token()
        body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
        self.post(self.SAML_GENERATION_ROUTE, body=body, expected_status=http.client.FORBIDDEN)

    def test_token_not_found(self):
        """Test that an invalid token in the request body raises an exception.

        Raises exception.TokenNotFound() - error Not Found 404

        """
        token_id = uuid.uuid4().hex
        body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
        self.post(self.SAML_GENERATION_ROUTE, body=body, expected_status=http.client.NOT_FOUND)

    def test_generate_ecp_route(self):
        """Test that the ECP generation endpoint produces XML.

        The ECP endpoint /v3/auth/OS-FEDERATION/saml2/ecp should take the same
        input as the SAML generation endpoint (scoped token ID + Service
        Provider ID).
        The controller should return a SAML assertion that is wrapped in a
        SOAP envelope.
        """
        self.config_fixture.config(group='saml', idp_entity_id=self.ISSUER)
        token_id = self._fetch_valid_token()
        body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
        with mock.patch.object(keystone_idp, '_sign_assertion', return_value=self.signed_assertion):
            http_response = self.post(self.ECP_GENERATION_ROUTE, body=body, response_content_type='text/xml', expected_status=http.client.OK)
        env_response = etree.fromstring(http_response.result)
        header = env_response[0]
        prefix = CONF.saml.relay_state_prefix
        self.assertThat(header[0].text, matchers.StartsWith(prefix))
        body = env_response[1]
        response = body[0]
        issuer = response[0]
        assertion = response[2]
        self.assertEqual(self.RECIPIENT, response.get('Destination'))
        self.assertEqual(self.ISSUER, issuer.text)
        user_attribute = assertion[4][0]
        self.assertIsInstance(user_attribute[0].text, str)
        user_domain_attribute = assertion[4][1]
        self.assertIsInstance(user_domain_attribute[0].text, str)
        role_attribute = assertion[4][2]
        self.assertIsInstance(role_attribute[0].text, str)
        project_attribute = assertion[4][3]
        self.assertIsInstance(project_attribute[0].text, str)
        project_domain_attribute = assertion[4][4]
        self.assertIsInstance(project_domain_attribute[0].text, str)
        group_attribute = assertion[4][5]
        self.assertIsInstance(group_attribute[0].text, str)

    @mock.patch('saml2.create_class_from_xml_string')
    @mock.patch('oslo_utils.fileutils.write_to_tempfile')
    @mock.patch.object(subprocess, 'check_output')
    def test_sign_assertion(self, check_output_mock, write_to_tempfile_mock, create_class_mock):
        write_to_tempfile_mock.return_value = 'tmp_path'
        check_output_mock.return_value = 'fakeoutput'
        keystone_idp._sign_assertion(self.signed_assertion)
        create_class_mock.assert_called_with(saml.Assertion, 'fakeoutput')

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

    @mock.patch('oslo_utils.fileutils.write_to_tempfile')
    @mock.patch.object(subprocess, 'check_output')
    def test_sign_assertion_fileutils_exc(self, check_output_mock, write_to_tempfile_mock):
        exception_msg = 'fake'
        write_to_tempfile_mock.side_effect = Exception(exception_msg)
        check_output_mock.return_value = '/usr/bin/xmlsec1'
        logger_fixture = self.useFixture(fixtures.LoggerFixture())
        self.assertRaises(exception.SAMLSigningError, keystone_idp._sign_assertion, self.signed_assertion)
        expected_log = 'Error when signing assertion, reason: %s\n' % exception_msg
        self.assertIn(expected_log, logger_fixture.output)

    def test_sign_assertion_logs_message_if_xmlsec1_is_not_installed(self):
        with mock.patch.object(subprocess, 'check_output') as co_mock:
            co_mock.side_effect = subprocess.CalledProcessError(returncode=1, cmd=CONF.saml.xmlsec1_binary)
            logger_fixture = self.useFixture(fixtures.LoggerFixture())
            self.assertRaises(exception.SAMLSigningError, keystone_idp._sign_assertion, self.signed_assertion)
            expected_log = 'Unable to locate xmlsec1 binary on the system. Check to make sure it is installed.\n'
            self.assertIn(expected_log, logger_fixture.output)