import bisect
import errno
import itertools
import os
import random
import stat
import sys
import time
import warnings
from .. import config, debug, errors, urlutils
from ..errors import LockError, ParamikoNotPresent, PathError, TransportError
from ..osutils import fancy_rename
from ..trace import mutter, warning
from ..transport import (ConnectedTransport, FileExists, FileFileStream,
def _asbytes_for_broken_paramiko(s):
    """Hacked asbytes() that does not raise Exception."""
    if not isinstance(s, bytes):
        encode = getattr(s, 'encode', None)
        if encode is not None:
            return encode('utf8')
        asbytes = getattr(s, 'asbytes', None)
        if asbytes is not None:
            return asbytes()
    return s