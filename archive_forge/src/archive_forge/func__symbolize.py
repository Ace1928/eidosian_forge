import importlib
import pkgutil
import re
import sys
import pbr.version
def _symbolize(mod_name, props):
    """Given a reference to a Python module object and an iterable of short
    string names for traits, registers symbols in the module corresponding to
    the full namespaced name for each trait.
    """
    for prop in props:
        leaf_mod = sys.modules[mod_name]
        value_base = '_'.join([m.upper() for m in mod_name.split('.')[1:]])
        value = value_base + '_' + prop.upper()
        setattr(THIS_LIB, value, value)
        setattr(leaf_mod, prop, value)