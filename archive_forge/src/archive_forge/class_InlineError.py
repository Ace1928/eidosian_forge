from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import copy
from pasta.base import ast_utils
from pasta.base import scope
class InlineError(Exception):
    pass