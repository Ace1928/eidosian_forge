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
def _getparams(self, params):
    result = {}
    for param in params:
        param, eq, value = param.partition('=')
        if not param.isalnum() or (eq and (not value)):
            return None
        result[param] = value if eq else True
    return result