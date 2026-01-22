import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def _set_loader(self, module: Any) -> None:

    class UNDEFINED:
        pass
    if getattr(module, '__loader__', UNDEFINED) in (None, self):
        try:
            module.__loader__ = self.loader
        except AttributeError:
            pass
    if getattr(module, '__spec__', None) is not None and getattr(module.__spec__, 'loader', None) is self:
        module.__spec__.loader = self.loader