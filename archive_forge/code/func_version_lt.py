import base64
import collections
import json
import os
import os.path
import shlex
import string
from datetime import datetime
from packaging.version import Version
from .. import errors
from ..constants import DEFAULT_HTTP_HOST
from ..constants import DEFAULT_UNIX_SOCKET
from ..constants import DEFAULT_NPIPE
from ..constants import BYTE_UNITS
from ..tls import TLSConfig
from urllib.parse import urlparse, urlunparse
def version_lt(v1, v2):
    return compare_version(v1, v2) > 0