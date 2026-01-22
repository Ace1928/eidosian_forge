from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class CommentedMapValuesView(CommentedMapView):
    __slots__ = ()

    def __contains__(self, value):
        for key in self._mapping:
            if value == self._mapping[key]:
                return True
        return False

    def __iter__(self):
        for key in self._mapping._keys():
            yield self._mapping[key]