import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def CompareProducts(x, y, remote_products):
    x_remote = x._properties['remoteRef']._properties['remoteGlobalIDString']
    y_remote = y._properties['remoteRef']._properties['remoteGlobalIDString']
    x_index = remote_products.index(x_remote)
    y_index = remote_products.index(y_remote)
    return cmp(x_index, y_index)