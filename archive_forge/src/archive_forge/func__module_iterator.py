import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def _module_iterator(root, recursive=True):
    """Iterate over modules.

    Parameters
    ----------
    root : module
        Root module or package to iterate from.
    recursive : bool
        ``True`` to iterate within subpackages.

    Yields
    ------
    module
        The modules found.
    """
    yield root
    stack = collections.deque((root,))
    while stack:
        package = stack.popleft()
        paths = getattr(package, '__path__', [])
        for path in paths:
            modules = pkgutil.iter_modules([path])
            for finder, name, is_package in modules:
                module_name = f'{package.__name__}.{name}'
                module = sys.modules.get(module_name, None)
                if module is None:
                    module = _load_module(finder, module_name)
                if is_package:
                    if recursive:
                        stack.append(module)
                        yield module
                else:
                    yield module