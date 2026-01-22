from __future__ import absolute_import, division, print_function
import errno
import os
import platform
import random
import re
import string
import filecmp
from ansible.module_utils.basic import AnsibleModule, get_distribution
from ansible.module_utils.six import iteritems
def _get_current_timezone(self, phase):
    """Lookup the current timezone via `systemsetup -gettimezone`."""
    if phase not in self.status:
        self.status[phase] = self.execute(self.systemsetup, '-gettimezone')
    return self.status[phase]