import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
Converts logical expressions to corresponding TF calls.