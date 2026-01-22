from __future__ import absolute_import, division, print_function
import abc
import datetime
import errno
import hashlib
import os
import re
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from .basic import (
@six.add_metaclass(abc.ABCMeta)
class OpenSSLObject(object):

    def __init__(self, path, state, force, check_mode):
        self.path = path
        self.state = state
        self.force = force
        self.name = os.path.basename(path)
        self.changed = False
        self.check_mode = check_mode

    def check(self, module, perms_required=True):
        """Ensure the resource is in its desired state."""

        def _check_state():
            return os.path.exists(self.path)

        def _check_perms(module):
            file_args = module.load_file_common_arguments(module.params)
            if module.check_file_absent_if_check_mode(file_args['path']):
                return False
            return not module.set_fs_attributes_if_different(file_args, False)
        if not perms_required:
            return _check_state()
        return _check_state() and _check_perms(module)

    @abc.abstractmethod
    def dump(self):
        """Serialize the object into a dictionary."""
        pass

    @abc.abstractmethod
    def generate(self):
        """Generate the resource."""
        pass

    def remove(self, module):
        """Remove the resource from the filesystem."""
        if self.check_mode:
            if os.path.exists(self.path):
                self.changed = True
            return
        try:
            os.remove(self.path)
            self.changed = True
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise OpenSSLObjectError(exc)
            else:
                pass