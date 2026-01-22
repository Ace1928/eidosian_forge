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
def _get_exclude_ips(self):
    exclude_hosts = self.module.params['exclude_hosts']
    exclude_ips = []
    if exclude_hosts is not None:
        for host in exclude_hosts:
            exclude_ips.extend(_convert_host_to_hex(host))
    return exclude_ips