import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def _exec_module(self, module: Any) -> None:
    self._set_loader(module)
    self.loader.exec_module(module)
    notify_module_loaded(module)