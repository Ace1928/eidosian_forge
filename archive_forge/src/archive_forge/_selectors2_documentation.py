import errno
import math
import select
import socket
import sys
import time
from collections import namedtuple
from ansible.module_utils.six.moves.collections_abc import Mapping
 Wrapper function for select.poll.poll() so that
            _syscall_wrapper can work with only seconds. 