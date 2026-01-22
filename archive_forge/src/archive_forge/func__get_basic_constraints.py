from __future__ import absolute_import, division, print_function
import abc
import binascii
import datetime
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def _get_basic_constraints(self):
    try:
        ext_keyusage_ext = self.cert.extensions.get_extension_for_class(x509.BasicConstraints)
        result = []
        result.append('CA:{0}'.format('TRUE' if ext_keyusage_ext.value.ca else 'FALSE'))
        if ext_keyusage_ext.value.path_length is not None:
            result.append('pathlen:{0}'.format(ext_keyusage_ext.value.path_length))
        return (sorted(result), ext_keyusage_ext.critical)
    except cryptography.x509.ExtensionNotFound:
        return (None, False)