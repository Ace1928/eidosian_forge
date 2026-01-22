from __future__ import absolute_import, division, print_function
import os
import pwd
import os.path
import tempfile
import re
import shlex
from operator import itemgetter
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url

    Add or remove key.
    