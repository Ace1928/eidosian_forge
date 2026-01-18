from __future__ import absolute_import, division, print_function
import filecmp
import os
import re
import shlex
import stat
import sys
import shutil
import tempfile
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.six import b, string_types
def write_ssh_wrapper(module):
    """
        This writes an shell wrapper for ssh options to be used with git
        this is only relevant for older versions of gitthat cannot
        handle the options themselves. Returns path to the script
    """
    try:
        if os.access(module.tmpdir, os.W_OK | os.R_OK | os.X_OK):
            fd, wrapper_path = tempfile.mkstemp(prefix=module.tmpdir + '/')
        else:
            raise OSError
    except (IOError, OSError):
        fd, wrapper_path = tempfile.mkstemp()
    template = b('#!/bin/sh\n%s $GIT_SSH_OPTS "$@"\n' % os.environ.get('GIT_SSH', os.environ.get('GIT_SSH_COMMAND', 'ssh')))
    with os.fdopen(fd, 'w+b') as fh:
        fh.write(template)
    st = os.stat(wrapper_path)
    os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)
    module.debug('Wrote temp git ssh wrapper (%s): %s' % (wrapper_path, template))
    module.add_cleanup_file(path=wrapper_path)
    return wrapper_path