import sys
import re
def version_tuple(vstring):
    components = []
    for x in _component_re.split(vstring):
        if x and x != '.':
            try:
                x = int(x)
            except ValueError:
                pass
            components.append(x)
    return tuple(components)