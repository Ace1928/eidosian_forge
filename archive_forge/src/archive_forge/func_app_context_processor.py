from __future__ import annotations
import os
import typing as t
from collections import defaultdict
from functools import update_wrapper
from .. import typing as ft
from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
from .scaffold import setupmethod
@setupmethod
def app_context_processor(self, f: T_template_context_processor) -> T_template_context_processor:
    """Like :meth:`context_processor`, but for templates rendered by every view, not
        only by the blueprint. Equivalent to :meth:`.Flask.context_processor`.
        """
    self.record_once(lambda s: s.app.template_context_processors.setdefault(None, []).append(f))
    return f