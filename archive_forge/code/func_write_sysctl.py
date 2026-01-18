from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def write_sysctl(self):
    fd, tmp_path = tempfile.mkstemp('.conf', '.ansible_m_sysctl_', os.path.dirname(self.sysctl_file))
    f = open(tmp_path, 'w')
    try:
        for l in self.fixed_lines:
            f.write(l.strip() + '\n')
    except IOError as e:
        self.module.fail_json(msg='Failed to write to file %s: %s' % (tmp_path, to_native(e)))
    f.flush()
    f.close()
    self.module.atomic_move(tmp_path, self.sysctl_file)