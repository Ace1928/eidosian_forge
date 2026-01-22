from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.common.parameters import env_fallback
class iDRACConnection:

    def __init__(self, module_params):
        if not HAS_OMSDK:
            raise ImportError('Dell OMSDK library is required for this module')
        self.idrac_ip = module_params['idrac_ip']
        self.idrac_user = module_params['idrac_user']
        self.idrac_pwd = module_params['idrac_password']
        self.idrac_port = module_params['idrac_port']
        if not all((self.idrac_ip, self.idrac_user, self.idrac_pwd)):
            raise ValueError('hostname, username and password required')
        self.handle = None
        self.creds = UserCredentials(self.idrac_user, self.idrac_pwd)
        self.validate_certs = module_params.get('validate_certs', False)
        self.ca_path = module_params.get('ca_path')
        verify_ssl = False
        if self.validate_certs is True:
            if self.ca_path is None:
                self.ca_path = self._get_omam_ca_env()
            verify_ssl = self.ca_path
        timeout = module_params.get('timeout', 30)
        if not timeout or not isinstance(timeout, int):
            timeout = 30
        self.pOp = WsManOptions(port=self.idrac_port, read_timeout=timeout, verify_ssl=verify_ssl)
        self.sdk = sdkinfra()
        if self.sdk is None:
            msg = 'Could not initialize iDRAC drivers.'
            raise RuntimeError(msg)

    def __enter__(self):
        self.idrac_ip = self.idrac_ip.strip('[]')
        self.sdk.importPath()
        protopref = ProtoPreference(ProtocolEnum.WSMAN)
        protopref.include_only(ProtocolEnum.WSMAN)
        self.handle = self.sdk.get_driver(self.sdk.driver_enum.iDRAC, self.idrac_ip, self.creds, protopref=protopref, pOptions=self.pOp)
        if self.handle is None:
            msg = 'Unable to communicate with iDRAC {0}. This may be due to one of the following: Incorrect username or password, unreachable iDRAC IP or a failure in TLS/SSL handshake.'.format(self.idrac_ip)
            raise RuntimeError(msg)
        return self.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handle.disconnect()
        return False

    def _get_omam_ca_env(self):
        """Check if the value is set in REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE or OMAM_CA_BUNDLE or True as ssl has to
        be validated from omsdk with single param and is default to false in omsdk"""
        return os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('CURL_CA_BUNDLE') or os.environ.get('OMAM_CA_BUNDLE') or True