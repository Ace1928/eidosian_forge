from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def _import_string_logic(dotted_path: str) -> Any:
    """Inspired by uvicorn — dotted paths should include a colon before the final item if that item is not a module.
    (This is necessary to distinguish between a submodule and an attribute when there is a conflict.).

    If the dotted path does not include a colon and the final item is not a valid module, importing as an attribute
    rather than a submodule will be attempted automatically.

    So, for example, the following values of `dotted_path` result in the following returned values:
    * 'collections': <module 'collections'>
    * 'collections.abc': <module 'collections.abc'>
    * 'collections.abc:Mapping': <class 'collections.abc.Mapping'>
    * `collections.abc.Mapping`: <class 'collections.abc.Mapping'> (though this is a bit slower than the previous line)

    An error will be raised under any of the following scenarios:
    * `dotted_path` contains more than one colon (e.g., 'collections:abc:Mapping')
    * the substring of `dotted_path` before the colon is not a valid module in the environment (e.g., '123:Mapping')
    * the substring of `dotted_path` after the colon is not an attribute of the module (e.g., 'collections:abc123')
    """
    from importlib import import_module
    components = dotted_path.strip().split(':')
    if len(components) > 2:
        raise ImportError(f"Import strings should have at most one ':'; received {dotted_path!r}")
    module_path = components[0]
    if not module_path:
        raise ImportError(f'Import strings should have a nonempty module name; received {dotted_path!r}')
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as e:
        if '.' in module_path:
            maybe_module_path, maybe_attribute = dotted_path.strip().rsplit('.', 1)
            try:
                return _import_string_logic(f'{maybe_module_path}:{maybe_attribute}')
            except ImportError:
                pass
            raise ImportError(f'No module named {module_path!r}') from e
        raise e
    if len(components) > 1:
        attribute = components[1]
        try:
            return getattr(module, attribute)
        except AttributeError as e:
            raise ImportError(f'cannot import name {attribute!r} from {module_path!r}') from e
    else:
        return module