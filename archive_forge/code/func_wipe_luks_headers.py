from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def wipe_luks_headers(device):
    wipe_offsets = []
    with open(device, 'rb') as f:
        data = f.read(LUKS_HEADER_L)
        if data == LUKS_HEADER:
            wipe_offsets.append(0)
        for offset in LUKS2_HEADER_OFFSETS:
            f.seek(offset)
            data = f.read(LUKS_HEADER_L)
            if data == LUKS2_HEADER2:
                wipe_offsets.append(offset)
    if wipe_offsets:
        with open(device, 'wb') as f:
            for offset in wipe_offsets:
                f.seek(offset)
                f.write(b'\x00\x00\x00\x00\x00\x00')