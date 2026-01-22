import re
import sys
class LooseVersion2(LooseVersion):
    """LooseVersion variant that restores Python 2 semantics

    In Python 2, comparing LooseVersions where paired components could be string
    and int always resulted in the string being "greater". In Python 3, this produced
    a TypeError.
    """

    def parse(self, vstring):
        self.vstring = vstring
        components = [x for x in self.component_re.split(vstring) if x and x != '.']
        for i, obj in enumerate(components):
            try:
                components[i] = _Py2Int(obj)
            except ValueError:
                pass
        self.version = components