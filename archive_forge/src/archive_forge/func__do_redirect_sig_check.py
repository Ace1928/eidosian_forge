import logging
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import time_util
from saml2.attribute_converter import to_local
from saml2.response import IncorrectlySigned
from saml2.s_utils import OtherError
from saml2.s_utils import VersionMismatch
from saml2.sigver import verify_redirect_signature
from saml2.validate import NotValid
from saml2.validate import valid_instance
def _do_redirect_sig_check(self, _saml_msg):
    issuer = self.sender()
    certs = self.sec.metadata.certs(issuer, 'any', 'signing')
    logger.debug('Certs to verify request sig: %s, _saml_msg: %s', certs, _saml_msg)
    verified = any((verify_redirect_signature(_saml_msg, self.sec.sec_backend, cert) for cert_name, cert in certs))
    logger.debug('Redirect request signature check: %s', verified)
    return verified