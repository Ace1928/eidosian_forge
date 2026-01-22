import abc
import base64
import hashlib
import os
import time
from urllib import parse as urlparse
import warnings
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import federation
def _generate_pkce_challenge(self):
    """Generate PKCE challenge string as defined in RFC 7636."""
    if self.code_challenge_method not in ('plain', 'S256'):
        raise exceptions.OidcGrantTypeMissmatch()
    self.code_verifier = self._generate_pkce_verifier()
    if self.code_challenge_method == 'plain':
        return self.code_verifier
    elif self.code_challenge_method == 'S256':
        _tmp = self.code_verifier.encode('ascii')
        _hash = hashlib.sha256(_tmp).digest()
        _tmp = base64.urlsafe_b64encode(_hash).decode('ascii')
        code_challenge = _tmp.rstrip('=')
        return code_challenge