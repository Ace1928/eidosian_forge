from __future__ import absolute_import, division, print_function
import glob
import json
import os
import platform
import re
import select
import shlex
import subprocess
import tempfile
import time
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.six import PY2, b
def write_to_override_file(file_name, file_contents):
    override_file = open(file_name, 'w')
    override_file.write(file_contents)
    override_file.close()