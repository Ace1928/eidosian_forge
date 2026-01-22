from __future__ import absolute_import, division, print_function
import os
import re
import time
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action.normal import ActionModule as _ActionModule
from ansible.utils.display import Display
from ansible.utils.hashing import checksum, checksum_s
def _check_dexec_eligibility(self, host):
    """Check if current python and task are eligble"""
    dexec = self.get_connection_option('import_modules')
    if dexec:
        display.vvvv('{prefix} enabled'.format(prefix=DEXEC_PREFIX), host)
        if not PY3:
            dexec = False
            display.vvvv('{prefix} disabled for when not Python 3'.format(prefix=DEXEC_PREFIX), host=host)
        if self._task.async_val:
            dexec = False
            display.vvvv('{prefix} disabled for a task using async'.format(prefix=DEXEC_PREFIX), host=host)
    else:
        display.vvvv('{prefix} disabled'.format(prefix=DEXEC_PREFIX), host)
        display.vvvv('{prefix} module execution time may be extended'.format(prefix=DEXEC_PREFIX), host)
    return dexec