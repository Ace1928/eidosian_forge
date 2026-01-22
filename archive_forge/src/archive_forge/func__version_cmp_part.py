import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
@classmethod
def _version_cmp_part(cls, va, vb):
    la = cls.re_all_digits_or_not.findall(va)
    lb = cls.re_all_digits_or_not.findall(vb)
    while la or lb:
        a = '0'
        b = '0'
        if la:
            a = la.pop(0)
        if lb:
            b = lb.pop(0)
        if cls.re_digits.match(a) and cls.re_digits.match(b):
            aval = int(a)
            bval = int(b)
            if aval < bval:
                return -1
            if aval > bval:
                return 1
        else:
            res = cls._version_cmp_string(a, b)
            if res != 0:
                return res
    return 0