from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class PySetObjectPtr(PyObjectPtr):
    _typename = 'PySetObject'

    @classmethod
    def _dummy_key(self):
        return gdb.lookup_global_symbol('_PySet_Dummy').value()

    def __iter__(self):
        dummy_ptr = self._dummy_key()
        table = self.field('table')
        for i in safe_range(self.field('mask') + 1):
            setentry = table[i]
            key = setentry['key']
            if key != 0 and key != dummy_ptr:
                yield PyObjectPtr.from_pyobject_ptr(key)

    def proxyval(self, visited):
        if self.as_address() in visited:
            return ProxyAlreadyVisited('%s(...)' % self.safe_tp_name())
        visited.add(self.as_address())
        members = (key.proxyval(visited) for key in self)
        if self.safe_tp_name() == 'frozenset':
            return frozenset(members)
        else:
            return set(members)

    def write_repr(self, out, visited):
        tp_name = self.safe_tp_name()
        if self.as_address() in visited:
            out.write('(...)')
            return
        visited.add(self.as_address())
        if not self.field('used'):
            out.write(tp_name)
            out.write('()')
            return
        if tp_name != 'set':
            out.write(tp_name)
            out.write('(')
        out.write('{')
        first = True
        for key in self:
            if not first:
                out.write(', ')
            first = False
            key.write_repr(out, visited)
        out.write('}')
        if tp_name != 'set':
            out.write(')')