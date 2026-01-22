import os
from io import BytesIO
from .. import (conflicts, errors, symbol_versioning, trace, transport,
from ..bzr import bzrdir
from ..bzr import conflicts as _mod_bzr_conflicts
from ..bzr import workingtree as bzrworkingtree
from ..bzr import workingtree_3, workingtree_4
from ..lock import write_locked
from ..lockdir import LockDir
from ..tree import TreeDirectory, TreeEntry, TreeFile, TreeLink
from . import TestCase, TestCaseWithTransport, TestSkipped
from .features import SymlinkFeature
class TestWorkingTreeFormatRegistry(TestCase):

    def setUp(self):
        super().setUp()
        self.registry = workingtree.WorkingTreeFormatRegistry()

    def test_register_unregister_format(self):
        format = SampleTreeFormat()
        self.registry.register(format)
        self.assertEqual(format, self.registry.get(b'Sample tree format.'))
        self.registry.remove(format)
        self.assertRaises(KeyError, self.registry.get, b'Sample tree format.')

    def test_get_all(self):
        format = SampleTreeFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra(self):
        format = SampleExtraTreeFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra_lazy(self):
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra_lazy('breezy.tests.test_workingtree', 'SampleExtraTreeFormat')
        formats = self.registry._get_all()
        self.assertEqual(1, len(formats))
        self.assertIsInstance(formats[0], SampleExtraTreeFormat)