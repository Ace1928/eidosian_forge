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
@__conn.setter
def __conn(self, value):
    warn("Setting __conn attribute on SMTPChannel is deprecated, set 'conn' instead", DeprecationWarning, 2)
    self.conn = value