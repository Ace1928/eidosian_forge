from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def yaml_key_comment_extend(self, key, comment, clear=False):
    r = self.ca._items.setdefault(key, [None, None, None, None])
    if clear or r[1] is None:
        if comment[1] is not None:
            assert isinstance(comment[1], list)
        r[1] = comment[1]
    else:
        r[1].extend(comment[0])
    r[0] = comment[0]