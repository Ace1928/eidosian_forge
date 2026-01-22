from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def add_exit_infos(self, msg=None, stdout=None, stderr=None):
    if msg:
        self._msgs.append(msg)
    if stdout:
        self._stdouts.append(stdout)
    if stderr:
        self._stderrs.append(stderr)