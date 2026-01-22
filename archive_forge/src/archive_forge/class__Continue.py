from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
class _Continue(object):

    def __init__(self):
        self.used = False
        self.control_var_name = None

    def __repr__(self):
        return '<_Continue(used: {}, var: {})>'.format(self.used, self.control_var_name)