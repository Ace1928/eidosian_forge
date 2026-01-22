from __future__ import annotations
import typing
import warnings
from .attr_map import AttrMap

        Call getattr on wrapped widget.  This has been the longstanding
        behaviour of AttrWrap, but is discouraged.  New code should be
        using AttrMap and .base_widget or .original_widget instead.
        