from __future__ import annotations
import logging # isort:skip
from ..subcommand import Subcommand
def _collect() -> list[type[Subcommand]]:
    from importlib import import_module
    from os import listdir
    from os.path import dirname
    results: list[type[Subcommand]] = []
    for file in listdir(dirname(__file__)):
        if not file.endswith('.py') or file in ('__init__.py', '__main__.py'):
            continue
        modname = file.rstrip('.py')
        mod = import_module('.' + modname, __package__)
        for name in dir(mod):
            attr = getattr(mod, name)
            try:
                if isinstance(attr, type) and issubclass(attr, Subcommand):
                    if not getattr(attr, 'name', None):
                        continue
                    results.append(attr)
            except TypeError:
                pass
    results = sorted(results, key=lambda attr: attr.name)
    return results