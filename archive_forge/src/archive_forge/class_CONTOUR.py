from __future__ import annotations
import functools
class CONTOUR(BuiltinFilter):
    name = 'Contour'
    filterargs = ((3, 3), 1, 255, (-1, -1, -1, -1, 8, -1, -1, -1, -1))