from __future__ import (absolute_import, division, print_function)
from os import path, walk
import re
import pathlib
import ansible.constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.action import ActionBase
from ansible.utils.vars import combine_vars
def _set_dir_defaults(self):
    if not self.depth:
        self.depth = 0
    if self.files_matching:
        self.matcher = re.compile('{0}'.format(self.files_matching))
    else:
        self.matcher = None
    if not self.ignore_files:
        self.ignore_files = list()
    if isinstance(self.ignore_files, string_types):
        self.ignore_files = self.ignore_files.split()
    elif isinstance(self.ignore_files, dict):
        return {'failed': True, 'message': '{0} must be a list'.format(self.ignore_files)}