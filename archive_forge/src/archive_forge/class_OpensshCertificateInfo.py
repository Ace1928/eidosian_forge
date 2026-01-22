from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@six.add_metaclass(abc.ABCMeta)
class OpensshCertificateInfo:
    """Encapsulates all certificate information which is signed by a CA key"""

    def __init__(self, nonce=None, serial=None, cert_type=None, key_id=None, principals=None, valid_after=None, valid_before=None, critical_options=None, extensions=None, reserved=None, signing_key=None):
        self.nonce = nonce
        self.serial = serial
        self._cert_type = cert_type
        self.key_id = key_id
        self.principals = principals
        self.valid_after = valid_after
        self.valid_before = valid_before
        self.critical_options = critical_options
        self.extensions = extensions
        self.reserved = reserved
        self.signing_key = signing_key
        self.type_string = None

    @property
    def cert_type(self):
        if self._cert_type == _USER_TYPE:
            return 'user'
        elif self._cert_type == _HOST_TYPE:
            return 'host'
        else:
            return ''

    @cert_type.setter
    def cert_type(self, cert_type):
        if cert_type == 'user' or cert_type == _USER_TYPE:
            self._cert_type = _USER_TYPE
        elif cert_type == 'host' or cert_type == _HOST_TYPE:
            self._cert_type = _HOST_TYPE
        else:
            raise ValueError('%s is not a valid certificate type' % cert_type)

    def signing_key_fingerprint(self):
        return fingerprint(self.signing_key)

    @abc.abstractmethod
    def public_key_fingerprint(self):
        pass

    @abc.abstractmethod
    def parse_public_numbers(self, parser):
        pass