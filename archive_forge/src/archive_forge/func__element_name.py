import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def _element_name(el):
    if isinstance(el, etree.CommentBase):
        return 'comment'
    elif isinstance(el, str):
        return 'string'
    else:
        return _nons(el.tag)