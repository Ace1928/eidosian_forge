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
class DHParameterBase(object):

    def __init__(self, module):
        self.state = module.params['state']
        self.path = module.params['path']
        self.size = module.params['size']
        self.force = module.params['force']
        self.changed = False
        self.return_content = module.params['return_content']
        self.backup = module.params['backup']
        self.backup_file = None

    @abc.abstractmethod
    def _do_generate(self, module):
        """Actually generate the DH params."""
        pass

    def generate(self, module):
        """Generate DH params."""
        changed = False
        if self.force or not self._check_params_valid(module):
            self._do_generate(module)
            changed = True
        if not self._check_fs_attributes(module):
            changed = True
        self.changed = changed

    def remove(self, module):
        if self.backup:
            self.backup_file = module.backup_local(self.path)
        try:
            os.remove(self.path)
            self.changed = True
        except OSError as exc:
            module.fail_json(msg=to_native(exc))

    def check(self, module):
        """Ensure the resource is in its desired state."""
        if self.force:
            return False
        return self._check_params_valid(module) and self._check_fs_attributes(module)

    @abc.abstractmethod
    def _check_params_valid(self, module):
        """Check if the params are in the correct state"""
        pass

    def _check_fs_attributes(self, module):
        """Checks (and changes if not in check mode!) fs attributes"""
        file_args = module.load_file_common_arguments(module.params)
        if module.check_file_absent_if_check_mode(file_args['path']):
            return False
        return not module.set_fs_attributes_if_different(file_args, False)

    def dump(self):
        """Serialize the object into a dictionary."""
        result = {'size': self.size, 'filename': self.path, 'changed': self.changed}
        if self.backup_file:
            result['backup_file'] = self.backup_file
        if self.return_content:
            content = load_file_if_exists(self.path, ignore_errors=True)
            result['dhparams'] = content.decode('utf-8') if content else None
        return result