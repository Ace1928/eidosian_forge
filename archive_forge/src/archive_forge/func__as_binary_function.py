import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
def _as_binary_function(self, func_name, arg1, arg2):
    return templates.replace_as_expression('func_name(arg1, arg2)', func_name=parser.parse_expression(func_name), arg1=arg1, arg2=arg2)