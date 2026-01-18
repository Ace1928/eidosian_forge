import collections
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (

  ::

    set(<variable> <value>
        [[CACHE <type> <docstring> [FORCE]] | PARENT_SCOPE])

  :see: https://cmake.org/cmake/help/v3.0/command/set.html?
  