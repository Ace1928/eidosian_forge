from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _generate_temp_certificate(self):
    key_copy = os.path.join(self.module.tmpdir, os.path.basename(self.public_key))
    try:
        self.module.preserved_copy(self.public_key, key_copy)
    except OSError as e:
        self.module.fail_json(msg='Unable to stage temporary key: %s' % to_native(e))
    self.module.add_cleanup_file(key_copy)
    self.ssh_keygen.generate_certificate(key_copy, self.identifier, self.options, self.pkcs11_provider, self.principals, self.serial_number, self.signature_algorithm, self.signing_key, self.type, self.time_parameters, self.use_agent, environ_update=dict(TZ='UTC'), check_rc=True)
    temp_cert = os.path.splitext(key_copy)[0] + '-cert.pub'
    self.module.add_cleanup_file(temp_cert)
    return temp_cert