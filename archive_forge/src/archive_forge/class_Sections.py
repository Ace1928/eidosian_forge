from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
class Sections(enum.Enum):
    ARGS = 0
    RETURNS = 1
    YIELDS = 2
    RAISES = 3
    TYPE = 4