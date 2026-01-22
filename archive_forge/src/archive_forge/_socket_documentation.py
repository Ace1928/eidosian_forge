import errno
import select
import socket
import six
import sys
from ._exceptions import *
from ._ssl_compat import *
from ._utils import *

    Return the global timeout setting(second) to connect.
    