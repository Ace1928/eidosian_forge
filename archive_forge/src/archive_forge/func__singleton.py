from __future__ import (absolute_import, division, print_function)
import re
def _singleton(name, constructor):
    if name in _singletons:
        return _singletons[name]
    _singletons[name] = constructor()
    return _singletons[name]