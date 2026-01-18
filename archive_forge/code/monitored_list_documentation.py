from __future__ import annotations
import functools
import typing
import warnings

        >>> ml = MonitoredFocusList([-2,0,1,-3,2,-1,3], focus=4)
        >>> ml.sort(); ml
        MonitoredFocusList([-3, -2, -1, 0, 1, 2, 3], focus=5)
        