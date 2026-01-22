import logging
from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (

  ``add_executable()`` has a couple of forms:

  * normal executables
  * imported executables
  * alias executables

  This function is just the dispatcher

  :see: https://cmake.org/cmake/help/latest/command/add_executable.html
  