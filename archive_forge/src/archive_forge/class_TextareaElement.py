import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class TextareaElement(InputMixin, HtmlElement):
    """
    ``<textarea>`` element.  You can get the name with ``.name`` and
    get/set the value with ``.value``
    """

    @property
    def value(self):
        """
        Get/set the value (which is the contents of this element)
        """
        content = self.text or ''
        if self.tag.startswith('{%s}' % XHTML_NAMESPACE):
            serialisation_method = 'xml'
        else:
            serialisation_method = 'html'
        for el in self:
            content += etree.tostring(el, method=serialisation_method, encoding='unicode')
        return content

    @value.setter
    def value(self, value):
        del self[:]
        self.text = value

    @value.deleter
    def value(self):
        self.text = ''
        del self[:]