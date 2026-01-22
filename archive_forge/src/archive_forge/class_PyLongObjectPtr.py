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
class PyLongObjectPtr(PyObjectPtr):
    _typename = 'PyLongObject'

    def proxyval(self, visited):
        """
        Python's Include/longobjrep.h has this declaration:
           struct _longobject {
               PyObject_VAR_HEAD
               digit ob_digit[1];
           };

        with this description:
            The absolute value of a number is equal to
                 SUM(for i=0 through abs(ob_size)-1) ob_digit[i] * 2**(SHIFT*i)
            Negative numbers are represented with ob_size < 0;
            zero is represented by ob_size == 0.

        where SHIFT can be either:
            #define PyLong_SHIFT        30
            #define PyLong_SHIFT        15
        """
        ob_size = long(self.field('ob_size'))
        if ob_size == 0:
            return 0
        ob_digit = self.field('ob_digit')
        if gdb.lookup_type('digit').sizeof == 2:
            SHIFT = 15
        else:
            SHIFT = 30
        digits = [long(ob_digit[i]) * 2 ** (SHIFT * i) for i in safe_range(abs(ob_size))]
        result = sum(digits)
        if ob_size < 0:
            result = -result
        return result

    def write_repr(self, out, visited):
        proxy = self.proxyval(visited)
        out.write('%s' % proxy)