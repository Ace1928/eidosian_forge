import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def _load_module(self, fullname: str) -> Any:
    module = self.loader.load_module(fullname)
    self._set_loader(module)
    notify_module_loaded(module)
    return module