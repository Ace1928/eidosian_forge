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
@__greeting.setter
def __greeting(self, value):
    warn("Setting __greeting attribute on SMTPChannel is deprecated, set 'seen_greeting' instead", DeprecationWarning, 2)
    self.seen_greeting = value