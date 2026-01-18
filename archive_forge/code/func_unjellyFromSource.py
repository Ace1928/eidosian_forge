import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def unjellyFromSource(stringOrFile):
    """
    Pass me a string of code or a filename that defines an 'app' variable (in
    terms of Abstract Objects!), and I'll execute it and unjelly the resulting
    AOT for you, returning a newly unpersisted Application object!
    """
    ns = {'Instance': Instance, 'InstanceMethod': InstanceMethod, 'Class': Class, 'Function': Function, 'Module': Module, 'Ref': Ref, 'Deref': Deref, 'Copyreg': Copyreg}
    if hasattr(stringOrFile, 'read'):
        source = stringOrFile.read()
    else:
        source = stringOrFile
    code = compile(source, '<source>', 'exec')
    eval(code, ns, ns)
    if 'app' in ns:
        return unjellyFromAOT(ns['app'])
    else:
        raise ValueError("%s needs to define an 'app', it didn't!" % stringOrFile)