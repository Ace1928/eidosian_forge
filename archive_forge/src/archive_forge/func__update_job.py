from __future__ import absolute_import, division, print_function
import os
import platform
import pwd
import re
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.six.moves import shlex_quote
def _update_job(self, name, job, addlinesfunction):
    ansiblename = self.do_comment(name)
    newlines = []
    comment = None
    for l in self.lines:
        if comment is not None:
            addlinesfunction(newlines, comment, job)
            comment = None
        elif l == ansiblename:
            comment = l
        else:
            newlines.append(l)
    self.lines = newlines
    if len(newlines) == 0:
        return True
    else:
        return False