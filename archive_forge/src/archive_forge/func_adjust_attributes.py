from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def adjust_attributes(token, replacements):
    needs_adjustment = viewkeys(token['data']) & viewkeys(replacements)
    if needs_adjustment:
        token['data'] = type(token['data'])(((replacements.get(k, k), v) for k, v in token['data'].items()))