from __future__ import (absolute_import, division, print_function)
import re
import shlex
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.become import BecomeBase
 checks if the expected password prompt exists in b_output 