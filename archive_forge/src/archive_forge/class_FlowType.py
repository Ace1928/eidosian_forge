from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
class FlowType(common.EnumObject):
    """
  Enumeration for flow control types
  """
    _id_map = {}