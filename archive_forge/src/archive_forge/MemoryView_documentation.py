from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode

        Slice a memoryviewslice.

        indices     - list of index nodes. If not a SliceNode, or NoneNode,
                      then it must be coercible to Py_ssize_t

        Simply call __pyx_memoryview_slice_memviewslice with the right
        arguments, unless the dimension is omitted or a bare ':', in which
        case we copy over the shape/strides/suboffsets attributes directly
        for that dimension.
        