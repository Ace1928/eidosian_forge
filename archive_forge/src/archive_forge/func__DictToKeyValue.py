from __future__ import absolute_import
from __future__ import print_function
from collections import namedtuple
import copy
import hashlib
import os
import six
def _DictToKeyValue(d):
    return ['%s=%s' % (k, d[k]) for k in sorted(d.keys())]