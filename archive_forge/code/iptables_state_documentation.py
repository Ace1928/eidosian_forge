from __future__ import absolute_import, division, print_function
import re
import os
import time
import tempfile
import filecmp
import shutil
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native

    Convert raw iptables-save output into usable datastructure, for reliable
    comparisons between initial and final states.
    