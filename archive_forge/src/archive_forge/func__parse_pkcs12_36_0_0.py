from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
import sys
import traceback
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse, ParseResult
from ._asn1 import serialize_asn1_string_as_der
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
from .basic import (
from ._objects import (
from ._obj2txt import obj2txt
def _parse_pkcs12_36_0_0(pkcs12_bytes, passphrase=None):
    pkcs12 = _load_pkcs12(pkcs12_bytes, passphrase)
    additional_certificates = [cert.certificate for cert in pkcs12.additional_certs]
    private_key = pkcs12.key
    certificate = None
    friendly_name = None
    if pkcs12.cert:
        certificate = pkcs12.cert.certificate
        friendly_name = pkcs12.cert.friendly_name
    return (private_key, certificate, additional_certificates, friendly_name)