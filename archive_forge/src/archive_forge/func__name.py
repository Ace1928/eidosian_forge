import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def _name(self):
    if self.get('name'):
        return self.get('name')
    elif self.get('id'):
        return '#' + self.get('id')
    iter_tags = self.body.iter
    forms = list(iter_tags('form'))
    if not forms:
        forms = list(iter_tags('{%s}form' % XHTML_NAMESPACE))
    return str(forms.index(self))