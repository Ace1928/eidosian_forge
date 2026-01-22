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
class DHParameterCryptography(DHParameterBase):

    def __init__(self, module):
        super(DHParameterCryptography, self).__init__(module)
        self.crypto_backend = cryptography.hazmat.backends.default_backend()

    def _do_generate(self, module):
        """Actually generate the DH params."""
        params = cryptography.hazmat.primitives.asymmetric.dh.generate_parameters(generator=2, key_size=self.size, backend=self.crypto_backend)
        result = params.parameter_bytes(encoding=cryptography.hazmat.primitives.serialization.Encoding.PEM, format=cryptography.hazmat.primitives.serialization.ParameterFormat.PKCS3)
        if self.backup:
            self.backup_file = module.backup_local(self.path)
        write_file(module, result)

    def _check_params_valid(self, module):
        """Check if the params are in the correct state"""
        try:
            with open(self.path, 'rb') as f:
                data = f.read()
            params = cryptography.hazmat.primitives.serialization.load_pem_parameters(data, backend=self.crypto_backend)
        except Exception as dummy:
            return False
        bits = count_bits(params.parameter_numbers().p)
        return bits == self.size