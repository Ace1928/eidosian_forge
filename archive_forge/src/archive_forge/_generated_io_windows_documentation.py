from __future__ import annotations
from typing import TYPE_CHECKING, ContextManager
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED
from ._run import GLOBAL_RUN_CONTEXT
import sys
TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__ and `#52
    <https://github.com/python-trio/trio/issues/52>`__.
    