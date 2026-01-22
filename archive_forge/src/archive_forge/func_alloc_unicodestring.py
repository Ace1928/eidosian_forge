import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version
def alloc_unicodestring(self, string, gdbvar=None):
    postfix = libpython.get_inferior_unicode_postfix()
    funcname = 'PyUnicode%s_DecodeUnicodeEscape' % (postfix,)
    data = string.encode('unicode_escape').decode('iso8859-1')
    return self.pyobject_fromcode('(PyObject *) %s("%s", %d, "strict")' % (funcname, data.replace('"', '\\"').replace('\\', '\\\\'), len(data)), gdbvar=gdbvar)