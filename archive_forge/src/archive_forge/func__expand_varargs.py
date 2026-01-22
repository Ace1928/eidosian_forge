import re
import copy
import inspect
import ast
import textwrap
def _expand_varargs(fn):
    fn_ast = function_to_ast(fn)
    fn_expanded_ast = expand_function_ast_varargs(fn_ast, expand_number)
    return function_ast_to_function(fn_expanded_ast, stacklevel=2)