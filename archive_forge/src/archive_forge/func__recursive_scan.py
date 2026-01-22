import importlib
import inspect
import pkgutil
import sys
from types import ModuleType
from typing import Dict
def _recursive_scan(package: ModuleType, dict_: Dict[str, str]) -> None:
    pkg_dir = package.__path__
    module_location = package.__name__
    for _module_loader, name, ispkg in pkgutil.iter_modules(pkg_dir):
        module_name = f'{module_location}.{name}'
        module = importlib.import_module(module_name)
        dict_[name] = inspect.getsource(module)
        if ispkg:
            _recursive_scan(module, dict_)