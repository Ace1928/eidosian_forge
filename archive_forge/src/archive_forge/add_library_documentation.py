import logging
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (

  ``add_library()`` has several forms:

  * normal libraires
  * imported libraries
  * object libraries
  * alias libraries
  * interface libraries

  This function is just the dispatcher

  :see: https://cmake.org/cmake/help/latest/command/add_library.html
  