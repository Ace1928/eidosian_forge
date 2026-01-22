import unittest
from traits.api import HasTraits, Instance, Str
class TestCopyTraitsSharedCopyNone(CopyTraits, CopyTraitsSharedCopyNone):
    __test__ = False

    def setUp(self):
        self.set_shared_copy('deep')