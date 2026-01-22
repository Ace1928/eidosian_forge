import importlib
import pkgutil
import re
import sys
import pbr.version
def _walk_submodules(package, recursive, callback, **kwargs):
    """Recursively walk the repository's submodules and invoke a callback for
    each module with the list of short trait names found therein.

    :param package: The package (name or module obj) to start from.
    :param recursive: If True, recurse depth-first.
    :param callback: Callable to be invoked for each module. The signature is::

            callback(mod_name, props, **kwargs)

        * mod_name: the string name of the module (e.g. 'os_traits.hw.cpu').
        * props: an iterable of short string names for traits, gleaned from the
          TRAITS member of that module, defaulting to [].
        * kwargs: The same kwargs as passed to _walk_submodules, useful for
          tracking data across calls.
    :param kwargs: Arbitrary keyword arguments to be passed to the callback on
        each invocation.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    for loader, mod_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        if TEST_DIR in mod_name:
            continue
        imported = importlib.import_module(mod_name)
        props = getattr(imported, 'TRAITS', [])
        callback(mod_name, props, **kwargs)
        if recursive and is_pkg:
            _walk_submodules(mod_name, recursive, callback, **kwargs)