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
def _set_rset_state(self):
    """Reset all state variables except the greeting."""
    self._set_post_data_state()
    self.received_data = ''
    self.received_lines = []