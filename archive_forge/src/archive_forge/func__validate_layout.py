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
def _validate_layout(self, stack_context, start_extent, end_extent):
    config = stack_context.config
    if end_extent[1] > config.format.linewidth:
        return False
    size = end_extent - start_extent
    if not self._wrap:
        if size[0] > 1 and len(self.name) > config.format.max_prefix_chars:
            return False
    return True