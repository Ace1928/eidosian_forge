from __future__ import absolute_import, division, print_function
import base64
import errno
import hashlib
import hmac
import os
import os.path
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native

    Transform a key, either taken from a known_host file or provided by the
    user, into a normalized form.
    The host part (which might include multiple hostnames or be hashed) gets
    replaced by the provided host. Also, any spurious information gets removed
    from the end (like the username@host tag usually present in hostkeys, but
    absent in known_hosts files)
    