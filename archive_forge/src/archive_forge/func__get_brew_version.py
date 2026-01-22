from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _get_brew_version(self):
    if self.brew_version:
        return self.brew_version
    cmd = [self.brew_path, '--version']
    rc, out, err = self.module.run_command(cmd, check_rc=True)
    version = out.split('\n')[0].split(' ')[1]
    self.brew_version = version
    return self.brew_version