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
@__data.setter
def __data(self, value):
    warn("Setting __data attribute on SMTPChannel is deprecated, set 'received_data' instead", DeprecationWarning, 2)
    self.received_data = value