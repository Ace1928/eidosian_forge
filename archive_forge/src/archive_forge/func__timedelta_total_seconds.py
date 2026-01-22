from __future__ import absolute_import, division, print_function
import binascii
import contextlib
import datetime
import errno
import math
import mmap
import os
import re
import select
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.compat.datetime import utcnow
def _timedelta_total_seconds(timedelta):
    return (timedelta.microseconds + 0.0 + (timedelta.seconds + timedelta.days * 24 * 3600) * 10 ** 6) / 10 ** 6