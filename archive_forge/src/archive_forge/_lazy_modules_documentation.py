import importlib
from types import ModuleType
from typing import Any, Callable, Optional, Union
Creates a new top-level lazy module or initializes a nested one.

        Args:
            module_name_or_module: Name of module to lazily import, or a module object
                for a nested lazy module.
            import_exc: Custom Exception to raise when an ``ImportError`` occurs. Will only
                be used by the top-level ``lazy_module`` instance, not nested modules
        