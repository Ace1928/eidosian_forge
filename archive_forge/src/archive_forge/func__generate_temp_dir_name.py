from __future__ import (absolute_import, division, print_function)
import os
import os.path
import random
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import text_type, string_types
from ansible.plugins import AnsiblePlugin
@staticmethod
def _generate_temp_dir_name():
    return 'ansible-tmp-%s-%s-%s' % (time.time(), os.getpid(), random.randint(0, 2 ** 48))