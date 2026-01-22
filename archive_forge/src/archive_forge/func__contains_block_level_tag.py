import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def _contains_block_level_tag(el):
    for el in el.iter(etree.Element):
        if _nons(el.tag) in defs.block_tags:
            return True
    return False