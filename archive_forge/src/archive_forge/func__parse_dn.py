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
def _parse_dn(name):
    """
    Parse a Distinguished Name.

    Can be of the form ``CN=Test, O = Something`` or ``CN = Test,O= Something``.
    """
    original_name = name
    name = name.lstrip()
    sep = b','
    if name.startswith(b'/'):
        sep = b'/'
        name = name[1:]
    result = []
    while name:
        try:
            attribute, name = _parse_dn_component(name, sep=sep)
        except OpenSSLObjectError as e:
            raise OpenSSLObjectError(u'Error while parsing distinguished name "{0}": {1}'.format(to_text(original_name), e))
        result.append(attribute)
        if name:
            if name[0:1] != sep or len(name) < 2:
                raise OpenSSLObjectError(u'Error while parsing distinguished name "{0}": unexpected end of string'.format(to_text(original_name)))
            name = name[1:]
    return result