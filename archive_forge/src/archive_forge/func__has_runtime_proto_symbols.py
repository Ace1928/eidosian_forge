import sys
import types
from typing import Tuple, Union
def _has_runtime_proto_symbols(mod: types.ModuleType) -> bool:
    return all((hasattr(mod, sym) for sym in _REQUIRED_SYMBOLS))