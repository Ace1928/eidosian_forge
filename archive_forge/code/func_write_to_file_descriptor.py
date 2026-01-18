from __future__ import (absolute_import, division, print_function)
import os
import hashlib
import json
import socket
import struct
import traceback
import uuid
from functools import partial
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import cPickle
def write_to_file_descriptor(fd, obj):
    """Handles making sure all data is properly written to file descriptor fd.

    In particular, that data is encoded in a character stream-friendly way and
    that all data gets written before returning.
    """
    src = cPickle.dumps(obj, protocol=0)
    src = src.replace(b'\r', b'\\r')
    data_hash = to_bytes(hashlib.sha1(src).hexdigest())
    os.write(fd, b'%d\n' % len(src))
    os.write(fd, src)
    os.write(fd, b'%s\n' % data_hash)