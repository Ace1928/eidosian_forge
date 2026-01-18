from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.compat import text_type
from ruamel.yaml.anchor import Anchor
def yaml_set_anchor(self, value, always_dump=False):
    self.anchor.value = value
    self.anchor.always_dump = always_dump