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
def _getaddr(self, arg):
    if not arg:
        return ('', '')
    if arg.lstrip().startswith('<'):
        address, rest = get_angle_addr(arg)
    else:
        address, rest = get_addr_spec(arg)
    if not address:
        return (address, rest)
    return (address.addr_spec, rest)