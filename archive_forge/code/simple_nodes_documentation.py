from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import NodeType, TreeNode

    Consume a 'cmake-format: [on|off]' comment
    