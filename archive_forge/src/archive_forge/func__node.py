import collections
import os
import re
import sys
import functools
import itertools
def _node(default=''):
    """ Helper to determine the node name of this machine.
    """
    try:
        import socket
    except ImportError:
        return default
    try:
        return socket.gethostname()
    except OSError:
        return default