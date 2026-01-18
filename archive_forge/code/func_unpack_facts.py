from __future__ import absolute_import, division, print_function
import binascii
import socket
import struct
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
@staticmethod
def unpack_facts(obj):
    result = dict(obj)
    if 'hardware-address' in result:
        result['hardware-address'] = to_native(unpack_mac(result[to_bytes('hardware-address')]))
    if 'ip-address' in result:
        result['ip-address'] = to_native(unpack_ip(result[to_bytes('ip-address')]))
    if 'hardware-type' in result:
        result['hardware-type'] = struct.unpack('!I', result[to_bytes('hardware-type')])
    return result