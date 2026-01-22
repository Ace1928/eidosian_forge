import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
class HtmlElementClassLookup(etree.CustomElementClassLookup):
    """A lookup scheme for HTML Element classes.

    To create a lookup instance with different Element classes, pass a tag
    name mapping of Element classes in the ``classes`` keyword argument and/or
    a tag name mapping of Mixin classes in the ``mixins`` keyword argument.
    The special key '*' denotes a Mixin class that should be mixed into all
    Element classes.
    """
    _default_element_classes = {}

    def __init__(self, classes=None, mixins=None):
        etree.CustomElementClassLookup.__init__(self)
        if classes is None:
            classes = self._default_element_classes.copy()
        if mixins:
            mixers = {}
            for name, value in mixins:
                if name == '*':
                    for n in classes.keys():
                        mixers.setdefault(n, []).append(value)
                else:
                    mixers.setdefault(name, []).append(value)
            for name, mix_bases in mixers.items():
                cur = classes.get(name, HtmlElement)
                bases = tuple(mix_bases + [cur])
                classes[name] = type(cur.__name__, bases, {})
        self._element_classes = classes

    def lookup(self, node_type, document, namespace, name):
        if node_type == 'element':
            return self._element_classes.get(name.lower(), HtmlElement)
        elif node_type == 'comment':
            return HtmlComment
        elif node_type == 'PI':
            return HtmlProcessingInstruction
        elif node_type == 'entity':
            return HtmlEntity
        return None