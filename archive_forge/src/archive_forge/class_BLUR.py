from __future__ import annotations
import functools
class BLUR(BuiltinFilter):
    name = 'Blur'
    filterargs = ((5, 5), 16, 0, (1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1))