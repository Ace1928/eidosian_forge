import importlib
import pkgutil
import re
import sys
import pbr.version
def _visualize(mod_name, props, seen=None):
    if mod_name in seen:
        return
    seen.add(mod_name)
    components = mod_name.split('.')
    tab = '   '
    indent = tab * (len(components) - 1)
    print('%s%s:' % (indent, components[-1].upper()))
    indent = tab * len(components)
    if props:
        print('%s%s' % (indent, ', '.join(props)))