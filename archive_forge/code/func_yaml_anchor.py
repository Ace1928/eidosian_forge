from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.compat import text_type
from ruamel.yaml.anchor import Anchor
def yaml_anchor(self, any=False):
    if not hasattr(self, Anchor.attrib):
        return None
    if any or self.anchor.always_dump:
        return self.anchor
    return None