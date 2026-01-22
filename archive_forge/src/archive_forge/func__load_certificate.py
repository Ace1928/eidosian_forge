from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _load_certificate(self):
    try:
        self.original_data = OpensshCertificate.load(self.path)
    except (TypeError, ValueError) as e:
        if self.regenerate in ('never', 'fail'):
            self.module.fail_json(msg='Unable to read existing certificate: %s' % to_native(e))
        self.module.warn('Unable to read existing certificate: %s' % to_native(e))