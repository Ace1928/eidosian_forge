from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splitpasswd(user):
    """splitpasswd('user:passwd') -> 'user', 'passwd'."""
    user, delim, passwd = user.partition(':')
    return (user, passwd if delim else None)