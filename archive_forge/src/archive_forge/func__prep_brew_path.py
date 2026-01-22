from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _prep_brew_path(self):
    if not self.module:
        self.brew_path = None
        self.failed = True
        self.message = 'AnsibleModule not set.'
        raise HomebrewCaskException(self.message)
    self.brew_path = self.module.get_bin_path('brew', required=True, opt_dirs=self.path)
    if not self.brew_path:
        self.brew_path = None
        self.failed = True
        self.message = 'Unable to locate homebrew executable.'
        raise HomebrewCaskException('Unable to locate homebrew executable.')
    return self.brew_path