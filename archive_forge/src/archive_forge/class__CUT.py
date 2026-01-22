import unittest
from zope.interface.tests import OptimizationTestMixin
class _CUT(BaseAdapterRegistry):

    class LookupClass:
        _changed = _extendors = ()

        def __init__(self, reg):
            pass

        def changed(self, orig):
            self._changed += (orig,)

        def add_extendor(self, provided):
            self._extendors += (provided,)

        def remove_extendor(self, provided):
            self._extendors = tuple([x for x in self._extendors if x != provided])