import re
import textwrap
from ast import literal_eval
from inspect import cleandoc
from weakref import WeakKeyDictionary
from parso.python import tree
from parso.cache import parser_cache
from parso import split_lines
def _function_is_x_method(*method_names):

    def wrapper(function_node):
        """
        This is a heuristic. It will not hold ALL the times, but it will be
        correct pretty much for anyone that doesn't try to beat it.
        staticmethod/classmethod are builtins and unless overwritten, this will
        be correct.
        """
        for decorator in function_node.get_decorators():
            dotted_name = decorator.children[1]
            if dotted_name.get_code() in method_names:
                return True
        return False
    return wrapper