import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
@__peer.setter
def __peer(self, value):
    warn("Setting __peer attribute on SMTPChannel is deprecated, set 'peer' instead", DeprecationWarning, 2)
    self.peer = value