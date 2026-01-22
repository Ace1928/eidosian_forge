from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class ServerSslProfilesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'alertTimeout': 'alert_timeout', 'allowExpiredCrl': 'allow_expired_crl', 'authenticate': 'authentication_frequency', 'authenticateDepth': 'authenticate_depth', 'authenticateName': 'authenticate_name', 'bypassOnClientCertFail': 'bypass_on_client_cert_fail', 'bypassOnHandshakeAlert': 'bypass_on_handshake_alert', 'c3dCaCert': 'c3d_ca_cert', 'c3dCaKey': 'c3d_ca_key', 'c3dCertExtensionIncludes': 'c3d_cert_extension_includes', 'c3dCertLifespan': 'c3d_cert_lifespan', 'caFile': 'ca_file', 'cacheSize': 'cache_size', 'cacheTimeout': 'cache_timeout', 'cipherGroup': 'cipher_group', 'crlFile': 'crl_file', 'defaultsFrom': 'parent', 'expireCertResponseControl': 'expire_cert_response_control', 'genericAlert': 'generic_alert', 'handshakeTimeout': 'handshake_timeout', 'maxActiveHandshakes': 'max_active_handshakes', 'modSslMethods': 'mod_ssl_methods', 'tmOptions': 'options', 'peerCertMode': 'peer_cert_mode', 'proxySsl': 'proxy_ssl', 'proxySslPassthrough': 'proxy_ssl_passthrough', 'renegotiatePeriod': 'renegotiate_period', 'renegotiateSize': 'renegotiate_size', 'retainCertificate': 'retain_certificate', 'secureRenegotiation': 'secure_renegotiation', 'serverName': 'server_name', 'sessionMirroring': 'session_mirroring', 'sessionTicket': 'session_ticket', 'sniDefault': 'sni_default', 'sniRequire': 'sni_require', 'sslC3d': 'ssl_c3d', 'sslForwardProxy': 'ssl_forward_proxy_enabled', 'sslForwardProxyBypass': 'ssl_forward_proxy_bypass', 'sslSignHash': 'ssl_sign_hash', 'strictResume': 'strict_resume', 'uncleanShutdown': 'unclean_shutdown', 'untrustedCertResponseControl': 'untrusted_cert_response_control'}
    returnables = ['full_path', 'name', 'parent', 'description', 'unclean_shutdown', 'strict_resume', 'ssl_forward_proxy_enabled', 'ssl_forward_proxy_bypass', 'sni_default', 'sni_require', 'ssl_c3d', 'session_mirroring', 'session_ticket', 'mod_ssl_methods', 'allow_expired_crl', 'retain_certificate', 'mode', 'bypass_on_client_cert_fail', 'bypass_on_handshake_alert', 'generic_alert', 'renegotiation', 'proxy_ssl', 'proxy_ssl_passthrough', 'peer_cert_mode', 'untrusted_cert_response_control', 'ssl_sign_hash', 'server_name', 'secure_renegotiation', 'renegotiate_size', 'renegotiate_period', 'options', 'ocsp', 'max_active_handshakes', 'key', 'handshake_timeout', 'expire_cert_response_control', 'cert', 'chain', 'authentication_frequency', 'ciphers', 'cipher_group', 'crl_file', 'cache_timeout', 'cache_size', 'ca_file', 'c3d_cert_lifespan', 'alert_timeout', 'c3d_ca_key', 'authenticate_depth', 'authenticate_name', 'c3d_ca_cert', 'c3d_cert_extension_includes']

    @property
    def c3d_cert_extension_includes(self):
        if self._values['c3d_cert_extension_includes'] is None:
            return None
        if len(self._values['c3d_cert_extension_includes']) == 0:
            return None
        self._values['c3d_cert_extension_includes'].sort()
        return self._values['c3d_cert_extension_includes']

    @property
    def options(self):
        if self._values['options'] is None:
            return None
        if len(self._values['options']) == 0:
            return None
        self._values['options'].sort()
        return self._values['options']

    @property
    def c3d_ca_cert(self):
        if self._values['c3d_ca_cert'] in [None, 'none']:
            return None
        return self._values['c3d_ca_cert']

    @property
    def ocsp(self):
        if self._values['ocsp'] in [None, 'none']:
            return None
        return self._values['ocsp']

    @property
    def server_name(self):
        if self._values['server_name'] in [None, 'none']:
            return None
        return self._values['server_name']

    @property
    def cipher_group(self):
        if self._values['cipher_group'] is None:
            return None
        if self._values['cipher_group'] == 'none':
            return 'none'
        return self._values['cipher_group']

    @property
    def authenticate_name(self):
        if self._values['authenticate_name'] in [None, 'none']:
            return None
        return self._values['authenticate_name']

    @property
    def c3d_ca_key(self):
        if self._values['c3d_ca_key'] in [None, 'none']:
            return None
        return self._values['c3d_ca_key']

    @property
    def ca_file(self):
        if self._values['ca_file'] in [None, 'none']:
            return None
        return self._values['ca_file']

    @property
    def crl_file(self):
        if self._values['crl_file'] in [None, 'none']:
            return None
        return self._values['crl_file']

    @property
    def authentication_frequency(self):
        if self._values['authentication_frequency'] in [None, 'none']:
            return None
        return self._values['authentication_frequency']

    @property
    def description(self):
        if self._values['description'] in [None, 'none']:
            return None
        return self._values['description']

    @property
    def proxy_ssl_passthrough(self):
        return flatten_boolean(self._values['proxy_ssl_passthrough'])

    @property
    def proxy_ssl(self):
        return flatten_boolean(self._values['proxy_ssl'])

    @property
    def generic_alert(self):
        return flatten_boolean(self._values['generic_alert'])

    @property
    def renegotiation(self):
        return flatten_boolean(self._values['renegotiation'])

    @property
    def bypass_on_handshake_alert(self):
        return flatten_boolean(self._values['bypass_on_handshake_alert'])

    @property
    def bypass_on_client_cert_fail(self):
        return flatten_boolean(self._values['bypass_on_client_cert_fail'])

    @property
    def mode(self):
        return flatten_boolean(self._values['mode'])

    @property
    def retain_certificate(self):
        return flatten_boolean(self._values['retain_certificate'])

    @property
    def allow_expired_crl(self):
        return flatten_boolean(self._values['allow_expired_crl'])

    @property
    def mod_ssl_methods(self):
        return flatten_boolean(self._values['mod_ssl_methods'])

    @property
    def session_ticket(self):
        return flatten_boolean(self._values['session_ticket'])

    @property
    def session_mirroring(self):
        return flatten_boolean(self._values['session_mirroring'])

    @property
    def unclean_shutdown(self):
        return flatten_boolean(self._values['unclean_shutdown'])

    @property
    def strict_resume(self):
        return flatten_boolean(self._values['strict_resume'])

    @property
    def ssl_forward_proxy_enabled(self):
        return flatten_boolean(self._values['ssl_forward_proxy_enabled'])

    @property
    def ssl_forward_proxy_bypass(self):
        return flatten_boolean(self._values['ssl_forward_proxy_bypass'])

    @property
    def sni_default(self):
        return flatten_boolean(self._values['sni_default'])

    @property
    def sni_require(self):
        return flatten_boolean(self._values['sni_require'])

    @property
    def ssl_c3d(self):
        return flatten_boolean(self._values['ssl_c3d'])