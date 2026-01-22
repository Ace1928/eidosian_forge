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
def __get_timezone(self):
    """ Return the current value of TZ= in /etc/environment """
    try:
        f = open('/etc/environment', 'r')
        etcenvironment = f.read()
        f.close()
    except Exception:
        self.module.fail_json(msg='Issue reading contents of /etc/environment')
    match = re.search('^TZ=(.*)$', etcenvironment, re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return None