from __future__ import absolute_import
from .Symtab import ModuleScope
from .PyrexTypes import *
from .UtilityCode import CythonUtilityCode
from .Errors import error
from .Scanning import StringSourceDescriptor
from . import MemoryView
from .StringEncoding import EncodedString

        Creates some entries for testing purposes and entries for
        cython.array() and for cython.view.*.
        