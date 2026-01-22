import ast
from typing import List, Optional, Tuple
from ._importlib import _resolve_name
def _absmodule(self, module_name: str, level: int) -> str:
    if level > 0:
        return _resolve_name(module_name, self.package, level)
    return module_name