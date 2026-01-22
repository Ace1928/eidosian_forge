from __future__ import annotations
import functools
class EMBOSS(BuiltinFilter):
    name = 'Emboss'
    filterargs = ((3, 3), 1, 128, (-1, 0, 0, 0, 1, 0, 0, 0, 0))