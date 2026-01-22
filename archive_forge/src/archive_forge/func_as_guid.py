import base64
import datetime
import re
import struct
import typing as t
import uuid
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
@per_sequence
def as_guid(value: t.Any) -> str:
    if isinstance(value, bytes):
        guid = uuid.UUID(bytes_le=value)
    else:
        b_value = base64.b64decode(str(value))
        guid = uuid.UUID(bytes_le=b_value)
    return str(guid)