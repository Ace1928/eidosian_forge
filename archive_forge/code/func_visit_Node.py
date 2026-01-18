from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def visit_Node(self, node):
    self.put(u'<???>')
    self.incomplete = True
    if self.description:
        warning(node.pos, 'Failed to convert code to string representation in {0}'.format(self.description), level=1)