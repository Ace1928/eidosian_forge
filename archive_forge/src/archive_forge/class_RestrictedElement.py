from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
class RestrictedElement(_etree.ElementBase):
    """A restricted Element class that filters out instances of some classes"""
    __slots__ = ()
    blacklist = _etree._Entity

    def _filter(self, iterator):
        blacklist = self.blacklist
        for child in iterator:
            if isinstance(child, blacklist):
                continue
            yield child

    def __iter__(self):
        iterator = super(RestrictedElement, self).__iter__()
        return self._filter(iterator)

    def iterchildren(self, tag=None, reversed=False):
        iterator = super(RestrictedElement, self).iterchildren(tag=tag, reversed=reversed)
        return self._filter(iterator)

    def iter(self, tag=None, *tags):
        iterator = super(RestrictedElement, self).iter(*tags, tag=tag)
        return self._filter(iterator)

    def iterdescendants(self, tag=None, *tags):
        iterator = super(RestrictedElement, self).iterdescendants(*tags, tag=tag)
        return self._filter(iterator)

    def itersiblings(self, tag=None, preceding=False):
        iterator = super(RestrictedElement, self).itersiblings(tag=tag, preceding=preceding)
        return self._filter(iterator)

    def getchildren(self):
        iterator = super(RestrictedElement, self).__iter__()
        return list(self._filter(iterator))

    def getiterator(self, tag=None):
        iterator = super(RestrictedElement, self).getiterator(tag)
        return self._filter(iterator)