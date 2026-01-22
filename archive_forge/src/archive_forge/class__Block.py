import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
class _Block(object):

    def __init__(self):
        self.is_function = False
        self.return_used = False
        self.create_guard_next = False
        self.create_guard_now = False

    def __repr__(self):
        return 'used: {}'.format(self.return_used)