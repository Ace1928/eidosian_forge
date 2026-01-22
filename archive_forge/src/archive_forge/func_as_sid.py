import base64
import datetime
import re
import struct
import typing as t
import uuid
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
@per_sequence
def as_sid(value: t.Any) -> str:
    if isinstance(value, bytes):
        view = memoryview(value)
    else:
        b_value = base64.b64decode(value)
        view = memoryview(b_value)
    if len(view) < 8:
        raise AnsibleFilterError('Raw SID bytes must be at least 8 bytes long')
    revision = view[0]
    sub_authority_count = view[1]
    authority = struct.unpack('>Q', view[:8])[0] & ~18446462598732840960
    view = view[8:]
    if len(view) < sub_authority_count * 4:
        raise AnsibleFilterError('Not enough data to unpack SID')
    sub_authorities: t.List[str] = []
    for dummy in range(sub_authority_count):
        auth = struct.unpack('<I', view[:4])[0]
        sub_authorities.append(str(auth))
        view = view[4:]
    return f'S-{revision}-{authority}-{'-'.join(sub_authorities)}'