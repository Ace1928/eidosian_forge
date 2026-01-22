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
class StackContext(object):
    """
  Aggregate information about the current stack. This object is passed down
  through all of the nested :code:`reflow()` function calls.
  """

    def __init__(self, config, first_token=None):
        self.config = config
        self.node_path = []
        self.first_token = first_token

    @contextlib.contextmanager
    def push_node(self, node):
        """
    Push `node` onto the `node_path` and yield a context manager. Pop `node`
    off of `node_path` when the context manager `__exit__()s`
    """
        self.node_path.append(node)
        yield None
        self.node_path.pop(-1)