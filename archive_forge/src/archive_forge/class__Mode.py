from __future__ import (absolute_import, division, print_function)
class _Mode(object):
    GARBAGE = 0
    KEY = 1
    EQUAL = 2
    IDENT_VALUE = 3
    QUOTED_VALUE = 4