from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
Provides a baseline of methods that a linker would implement.

    In every case this provides a "no" or "empty" answer. If a compiler
    implements any of these it needs a different mixin or to override that
    functionality itself.
    