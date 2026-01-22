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
class PyDictObjectPtr(PyObjectPtr):
    """
    Class wrapping a gdb.Value that's a PyDictObject* i.e. a dict instance
    within the process being debugged.
    """
    _typename = 'PyDictObject'

    def iteritems(self):
        """
        Yields a sequence of (PyObjectPtr key, PyObjectPtr value) pairs,
        analogous to dict.iteritems()
        """
        keys = self.field('ma_keys')
        values = self.field('ma_values')
        entries, nentries = self._get_entries(keys)
        for i in safe_range(nentries):
            ep = entries[i]
            if long(values):
                pyop_value = PyObjectPtr.from_pyobject_ptr(values[i])
            else:
                pyop_value = PyObjectPtr.from_pyobject_ptr(ep['me_value'])
            if not pyop_value.is_null():
                pyop_key = PyObjectPtr.from_pyobject_ptr(ep['me_key'])
                yield (pyop_key, pyop_value)

    def proxyval(self, visited):
        if self.as_address() in visited:
            return ProxyAlreadyVisited('{...}')
        visited.add(self.as_address())
        result = {}
        for pyop_key, pyop_value in self.iteritems():
            proxy_key = pyop_key.proxyval(visited)
            proxy_value = pyop_value.proxyval(visited)
            result[proxy_key] = proxy_value
        return result

    def write_repr(self, out, visited):
        if self.as_address() in visited:
            out.write('{...}')
            return
        visited.add(self.as_address())
        out.write('{')
        first = True
        for pyop_key, pyop_value in self.iteritems():
            if not first:
                out.write(', ')
            first = False
            pyop_key.write_repr(out, visited)
            out.write(': ')
            pyop_value.write_repr(out, visited)
        out.write('}')

    def _get_entries(self, keys):
        dk_nentries = int(keys['dk_nentries'])
        dk_size = int(keys['dk_size'])
        try:
            return (keys['dk_entries'], dk_size)
        except RuntimeError:
            pass
        if dk_size <= 255:
            offset = dk_size
        elif dk_size <= 65535:
            offset = 2 * dk_size
        elif dk_size <= 4294967295:
            offset = 4 * dk_size
        else:
            offset = 8 * dk_size
        ent_addr = keys['dk_indices'].address
        ent_addr = ent_addr.cast(_type_unsigned_char_ptr()) + offset
        ent_ptr_t = gdb.lookup_type('PyDictKeyEntry').pointer()
        ent_addr = ent_addr.cast(ent_ptr_t)
        return (ent_addr, dk_nentries)