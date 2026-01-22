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
def _make_string(self):
    from .AutoDocTransforms import AnnotationWriter
    writer = AnnotationWriter(description='Dataclass field')
    string = writer.write(self.arg)
    return ExprNodes.StringNode(self.pos, value=EncodedString(string))