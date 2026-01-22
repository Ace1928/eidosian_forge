from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
class WriteContext(object):
    """
  Global state for the writing functions
  """

    def __init__(self, config, infile_content):
        self.offswitch_location = None
        if sys.version_info[0] < 3:
            assert isinstance(infile_content, unicode)
        self.infile = io.BytesIO(bytearray(infile_content, 'utf-8'))
        self.outfile = CursorFile(config)

    def is_active(self):
        return self.offswitch_location is None