from types import ModuleType
from typing import Any
from .._mangling import is_mangled

    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    