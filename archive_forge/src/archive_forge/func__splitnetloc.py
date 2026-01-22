from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splitnetloc(url, start=0):
    delim = len(url)
    for c in '/?#':
        wdelim = url.find(c, start)
        if wdelim >= 0:
            delim = min(delim, wdelim)
    return (url[start:delim], url[delim:])