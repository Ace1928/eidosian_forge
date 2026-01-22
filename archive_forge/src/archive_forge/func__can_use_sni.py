import errno
import os
import socket
import sys
import six
from ._exceptions import *
from ._logging import *
from ._socket import*
from ._ssl_compat import *
from ._url import *
def _can_use_sni():
    return six.PY2 and sys.version_info >= (2, 7, 9) or sys.version_info >= (3, 2)