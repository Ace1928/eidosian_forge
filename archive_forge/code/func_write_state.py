from __future__ import absolute_import, division, print_function
import re
import os
import time
import tempfile
import filecmp
import shutil
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def write_state(b_path, lines, changed):
    """
    Write given contents to the given path, and return changed status.
    """
    tmpfd, tmpfile = tempfile.mkstemp()
    with os.fdopen(tmpfd, 'w') as f:
        f.write('{0}\n'.format('\n'.join(lines)))
    if not os.path.exists(b_path):
        b_destdir = os.path.dirname(b_path)
        destdir = to_native(b_destdir, errors='surrogate_or_strict')
        if b_destdir and (not os.path.exists(b_destdir)) and (not module.check_mode):
            try:
                os.makedirs(b_destdir)
            except Exception as err:
                module.fail_json(msg='Error creating %s: %s' % (destdir, to_native(err)), initial_state=lines)
        changed = True
    elif not filecmp.cmp(tmpfile, b_path):
        changed = True
    if changed and (not module.check_mode):
        try:
            shutil.copyfile(tmpfile, b_path)
        except Exception as err:
            path = to_native(b_path, errors='surrogate_or_strict')
            module.fail_json(msg='Error saving state into %s: %s' % (path, to_native(err)), initial_state=lines)
    return changed