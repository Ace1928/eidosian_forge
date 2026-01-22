from __future__ import (absolute_import, division, print_function)
import os
import subprocess
from subprocess import PIPE
from subprocess import Popen
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display

    USAGE:

    