from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import read_file
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
class CryptographyChainMatcher(ChainMatcher):

    @staticmethod
    def _parse_key_identifier(key_identifier, name, criterium_idx, module):
        if key_identifier:
            try:
                return binascii.unhexlify(key_identifier.replace(':', ''))
            except Exception:
                if criterium_idx is None:
                    module.warn('Criterium has invalid {0} value. Ignoring criterium.'.format(name))
                else:
                    module.warn('Criterium {0} in select_chain has invalid {1} value. Ignoring criterium.'.format(criterium_idx, name))
        return None

    def __init__(self, criterium, module):
        self.criterium = criterium
        self.test_certificates = criterium.test_certificates
        self.subject = []
        self.issuer = []
        if criterium.subject:
            self.subject = [(cryptography_name_to_oid(k), to_native(v)) for k, v in parse_name_field(criterium.subject, 'subject')]
        if criterium.issuer:
            self.issuer = [(cryptography_name_to_oid(k), to_native(v)) for k, v in parse_name_field(criterium.issuer, 'issuer')]
        self.subject_key_identifier = CryptographyChainMatcher._parse_key_identifier(criterium.subject_key_identifier, 'subject_key_identifier', criterium.index, module)
        self.authority_key_identifier = CryptographyChainMatcher._parse_key_identifier(criterium.authority_key_identifier, 'authority_key_identifier', criterium.index, module)

    def _match_subject(self, x509_subject, match_subject):
        for oid, value in match_subject:
            found = False
            for attribute in x509_subject:
                if attribute.oid == oid and value == to_native(attribute.value):
                    found = True
                    break
            if not found:
                return False
        return True

    def match(self, certificate):
        """
        Check whether an alternate chain matches the specified criterium.
        """
        chain = certificate.chain
        if self.test_certificates == 'last':
            chain = chain[-1:]
        elif self.test_certificates == 'first':
            chain = chain[:1]
        for cert in chain:
            try:
                x509 = cryptography.x509.load_pem_x509_certificate(to_bytes(cert), cryptography.hazmat.backends.default_backend())
                matches = True
                if not self._match_subject(x509.subject, self.subject):
                    matches = False
                if not self._match_subject(x509.issuer, self.issuer):
                    matches = False
                if self.subject_key_identifier:
                    try:
                        ext = x509.extensions.get_extension_for_class(cryptography.x509.SubjectKeyIdentifier)
                        if self.subject_key_identifier != ext.value.digest:
                            matches = False
                    except cryptography.x509.ExtensionNotFound:
                        matches = False
                if self.authority_key_identifier:
                    try:
                        ext = x509.extensions.get_extension_for_class(cryptography.x509.AuthorityKeyIdentifier)
                        if self.authority_key_identifier != ext.value.key_identifier:
                            matches = False
                    except cryptography.x509.ExtensionNotFound:
                        matches = False
                if matches:
                    return True
            except Exception as e:
                self.module.warn('Error while loading certificate {0}: {1}'.format(cert, e))
        return False