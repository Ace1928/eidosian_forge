from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
class DocstringInfo(collections.namedtuple('DocstringInfo', ('summary', 'description', 'args', 'returns', 'yields', 'raises'))):
    pass