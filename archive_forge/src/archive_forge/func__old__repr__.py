from __future__ import annotations
import sys
import copy
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.compat import MutableSliceableSequence, nprintf  # NOQA
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
from ruamel.yaml.tag import Tag
from collections.abc import MutableSet, Sized, Set, Mapping
def _old__repr__(self) -> str:
    if bool(self._post):
        end = ',\n  end=' + str(self._post)
    else:
        end = ''
    try:
        ln = max([len(str(k)) for k in self._items]) + 1
    except ValueError:
        ln = ''
    it = '    '.join([f'{str(k) + ':':{ln}} {v}\n' for k, v in self._items.items()])
    if it:
        it = '\n    ' + it + '  '
    return f'Comment(\n  start={self.comment},\n  items={{{it}}}{end})'