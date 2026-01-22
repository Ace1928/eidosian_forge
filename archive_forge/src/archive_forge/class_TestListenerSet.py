import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestListenerSet(TestListener):

    def setUp(self):
        super().setUp()
        self.cmd = listener.SetListener(self.app, None)

    def test_listener_set(self):
        arglist = [self._listener.id, '--name', 'new_name', '--sni-container-refs', self._listener.sni_container_refs[0], self._listener.sni_container_refs[1], '--default-tls-container-ref', self._listener.default_tls_container_ref, '--client-ca-tls-container-ref', self._listener.client_ca_tls_container_ref, '--client-authentication', self._listener.client_authentication, '--client-crl-container-ref', self._listener.client_crl_container_ref, '--allowed-cidr', self._listener.allowed_cidrs[0], '--allowed-cidr', self._listener.allowed_cidrs[1], '--tls-ciphers', self._listener.tls_ciphers, '--tls-version', self._listener.tls_versions[0], '--tls-version', self._listener.tls_versions[1], '--alpn-protocol', self._listener.alpn_protocols[0], '--alpn-protocol', self._listener.alpn_protocols[1], '--hsts-max-age', '15000000', '--hsts-include-subdomains', '--hsts-preload']
        verifylist = [('listener', self._listener.id), ('name', 'new_name'), ('sni_container_refs', self._listener.sni_container_refs), ('default_tls_container_ref', self._listener.default_tls_container_ref), ('client_ca_tls_container_ref', self._listener.client_ca_tls_container_ref), ('client_authentication', self._listener.client_authentication), ('client_crl_container_ref', self._listener.client_crl_container_ref), ('allowed_cidrs', self._listener.allowed_cidrs), ('tls_ciphers', self._listener.tls_ciphers), ('tls_versions', self._listener.tls_versions), ('alpn_protocols', self._listener.alpn_protocols), ('hsts_max_age', 15000000), ('hsts_include_subdomains', True), ('hsts_preload', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_with(self._listener.id, json={'listener': {'name': 'new_name', 'sni_container_refs': self._listener.sni_container_refs, 'default_tls_container_ref': self._listener.default_tls_container_ref, 'client_ca_tls_container_ref': self._listener.client_ca_tls_container_ref, 'client_authentication': self._listener.client_authentication, 'client_crl_container_ref': self._listener.client_crl_container_ref, 'allowed_cidrs': self._listener.allowed_cidrs, 'tls_ciphers': self._listener.tls_ciphers, 'tls_versions': self._listener.tls_versions, 'alpn_protocols': self._listener.alpn_protocols, 'hsts_max_age': 15000000, 'hsts_include_subdomains': True, 'hsts_preload': True}})

    def test_listener_set_suppressed(self):
        arglist = [self._listener.id, '--name', 'foo']
        verifylist = [('name', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertNotIn('hsts_preload', parsed_args)
        self.assertNotIn('hsts_include_subdomain', parsed_args)
        self.assertNotIn('hsts_max_age', parsed_args)
        self.api_mock.listener_set.assert_called_with(self._listener.id, json={'listener': {'name': 'foo'}})

    @mock.patch('osc_lib.utils.wait_for_status')
    def test_listener_set_wait(self, mock_wait):
        arglist = [self._listener.id, '--name', 'new_name', '--wait']
        verifylist = [('listener', self._listener.id), ('name', 'new_name'), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_with(self._listener.id, json={'listener': {'name': 'new_name'}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._listener.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_listener_set_tag(self):
        self.api_mock.listener_show.return_value = {'tags': ['foo']}
        arglist = [self._listener.id, '--tag', 'bar']
        verifylist = [('listener', self._listener.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_once()
        kwargs = self.api_mock.listener_set.mock_calls[0][2]
        tags = kwargs['json']['listener']['tags']
        self.assertEqual(2, len(tags))
        self.assertIn('foo', tags)
        self.assertIn('bar', tags)

    def test_listener_set_tag_no_tag(self):
        self.api_mock.listener_show.return_value = {'tags': ['foo']}
        arglist = [self._listener.id, '--tag', 'bar', '--no-tag']
        verifylist = [('listener', self._listener.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_once_with(self._listener.id, json={'listener': {'tags': ['bar']}})