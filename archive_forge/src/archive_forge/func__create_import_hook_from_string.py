import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def _create_import_hook_from_string(name: str) -> Callable:

    def import_hook(module: Any) -> Callable:
        module_name, function = name.split(':')
        attrs = function.split('.')
        __import__(module_name)
        callback = sys.modules[module_name]
        for attr in attrs:
            callback = getattr(callback, attr)
        return callback(module)
    return import_hook