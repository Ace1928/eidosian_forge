from __future__ import (absolute_import, division, print_function)
import re
import traceback
from collections.abc import Sequence
from ansible.errors.yaml_strings import (
from ansible.module_utils.common.text.converters import to_native, to_text
class AnsibleFileNotFound(AnsibleRuntimeError):
    """ a file missing failure """

    def __init__(self, message='', obj=None, show_content=True, suppress_extended_error=False, orig_exc=None, paths=None, file_name=None):
        self.file_name = file_name
        self.paths = paths
        if message:
            message += '\n'
        if self.file_name:
            message += "Could not find or access '%s'" % to_text(self.file_name)
        else:
            message += 'Could not find file'
        if self.paths and isinstance(self.paths, Sequence):
            searched = to_text('\n\t'.join(self.paths))
            if message:
                message += '\n'
            message += 'Searched in:\n\t%s' % searched
        message += ' on the Ansible Controller.\nIf you are using a module and expect the file to exist on the remote, see the remote_src option'
        super(AnsibleFileNotFound, self).__init__(message=message, obj=obj, show_content=show_content, suppress_extended_error=suppress_extended_error, orig_exc=orig_exc)