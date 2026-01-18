import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def unjellyFromAOT(aot):
    """
    Pass me an Abstract Object Tree, and I'll unjelly it for you.
    """
    return AOTUnjellier().unjelly(aot)