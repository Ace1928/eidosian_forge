from __future__ import absolute_import, division, print_function
import abc
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
def _do_generate(self, module):
    """Actually generate the DH params."""
    params = cryptography.hazmat.primitives.asymmetric.dh.generate_parameters(generator=2, key_size=self.size, backend=self.crypto_backend)
    result = params.parameter_bytes(encoding=cryptography.hazmat.primitives.serialization.Encoding.PEM, format=cryptography.hazmat.primitives.serialization.ParameterFormat.PKCS3)
    if self.backup:
        self.backup_file = module.backup_local(self.path)
    write_file(module, result)