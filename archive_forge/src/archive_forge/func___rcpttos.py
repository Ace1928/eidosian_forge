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
@__rcpttos.setter
def __rcpttos(self, value):
    warn("Setting __rcpttos attribute on SMTPChannel is deprecated, set 'rcpttos' instead", DeprecationWarning, 2)
    self.rcpttos = value