import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
def _as_unary_function(self, func_name, arg):
    return templates.replace_as_expression('func_name(arg)', func_name=parser.parse_expression(func_name), arg=arg)