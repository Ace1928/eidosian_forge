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
def _allow_ioerror(self, err, key):
    if err.errno != errno.ENOENT:
        return False
    return self.allow_no_file.get(key, False)