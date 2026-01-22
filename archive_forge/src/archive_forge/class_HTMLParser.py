import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class HTMLParser(etree.HTMLParser):
    """An HTML parser that is configured to return lxml.html Element
    objects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_element_class_lookup(HtmlElementClassLookup())