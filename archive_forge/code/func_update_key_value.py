from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def update_key_value(self, key):
    if key in self._ok:
        return
    for v in self.merge:
        if key in v[1]:
            ordereddict.__setitem__(self, key, v[1][key])
            return
    ordereddict.__delitem__(self, key)