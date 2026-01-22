from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
class RenameImport(object):
    """
    A class for import hooks mapping Py3 module names etc. to the Py2 equivalents.
    """
    RENAMER = True

    def __init__(self, old_to_new):
        """
        Pass in a dictionary-like object mapping from old names to new
        names. E.g. {'ConfigParser': 'configparser', 'cPickle': 'pickle'}
        """
        self.old_to_new = old_to_new
        both = set(old_to_new.keys()) & set(old_to_new.values())
        assert len(both) == 0 and len(set(old_to_new.values())) == len(old_to_new.values()), 'Ambiguity in renaming (handler not implemented)'
        self.new_to_old = dict(((new, old) for old, new in old_to_new.items()))

    def find_module(self, fullname, path=None):
        new_base_names = set([s.split('.')[0] for s in self.new_to_old])
        if fullname in new_base_names:
            return self
        return None

    def load_module(self, name):
        path = None
        if name in sys.modules:
            return sys.modules[name]
        elif name in self.new_to_old:
            oldname = self.new_to_old[name]
            module = self._find_and_load_module(oldname)
        else:
            module = self._find_and_load_module(name)
        sys.modules[name] = module
        return module

    def _find_and_load_module(self, name, path=None):
        """
        Finds and loads it. But if there's a . in the name, handles it
        properly.
        """
        bits = name.split('.')
        while len(bits) > 1:
            packagename = bits.pop(0)
            package = self._find_and_load_module(packagename, path)
            try:
                path = package.__path__
            except AttributeError:
                flog.debug('Package {0} has no __path__.'.format(package))
                if name in sys.modules:
                    return sys.modules[name]
                flog.debug('What to do here?')
        name = bits[0]
        module_info = imp.find_module(name, path)
        return imp.load_module(name, *module_info)