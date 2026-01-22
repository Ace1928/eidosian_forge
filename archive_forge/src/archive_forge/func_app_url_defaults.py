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
def app_url_defaults(self, f: T_url_defaults) -> T_url_defaults:
    """Like :meth:`url_defaults`, but for every request, not only those handled by
        the blueprint. Equivalent to :meth:`.Flask.url_defaults`.
        """
    self.record_once(lambda s: s.app.url_default_functions.setdefault(None, []).append(f))
    return f