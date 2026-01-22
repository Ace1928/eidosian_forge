from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def _load_builtins(self):
    from numba.core.typing import builtins, arraydecl, npdatetime
    from numba.core.typing import ctypes_utils, bufproto
    from numba.core.unsafe import eh
    self.install_registry(templates.builtin_registry)