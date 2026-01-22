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
def app_errorhandler(self, code: type[Exception] | int) -> t.Callable[[T_error_handler], T_error_handler]:
    """Like :meth:`errorhandler`, but for every request, not only those handled by
        the blueprint. Equivalent to :meth:`.Flask.errorhandler`.
        """

    def decorator(f: T_error_handler) -> T_error_handler:

        def from_blueprint(state: BlueprintSetupState) -> None:
            state.app.errorhandler(code)(f)
        self.record_once(from_blueprint)
        return f
    return decorator