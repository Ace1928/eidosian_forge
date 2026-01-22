from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
class ArgInfo(collections.namedtuple('ArgInfo', ('name', 'type', 'description'))):
    pass