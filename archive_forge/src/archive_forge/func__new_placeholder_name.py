from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
def _new_placeholder_name(self, field_names):
    while True:
        name = 'DATACLASS_PLACEHOLDER_%d' % self._placeholder_count
        if name not in self.placeholders and name not in field_names:
            break
        self._placeholder_count += 1
    return name