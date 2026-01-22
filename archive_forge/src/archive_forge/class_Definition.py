import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
class Definition(object):
    """Definition objects describe a unique definition of a variable.

  Subclasses of this may be used by passing an appropriate factory function to
  resolve.

  Attributes:
    param_of: Optional[ast.AST]
    directives: Dict, optional definition annotations
  """

    def __init__(self):
        self.param_of = None
        self.directives = {}

    def __repr__(self):
        return '%s[%d]' % (self.__class__.__name__, id(self))