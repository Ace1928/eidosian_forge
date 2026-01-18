import bz2
import os
import sys
import tempfile
from io import BytesIO
from ... import diff, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...tests import features, test_commit
from ...tree import InterTree
from .. import bzrdir, inventory, knitrepo
from ..bundle.apply_bundle import install_bundle, merge_bundle
from ..bundle.bundle_data import BundleTree
from ..bundle.serializer import read_bundle, v4, v09, write_bundle
from ..bundle.serializer.v4 import BundleSerializerV4
from ..bundle.serializer.v08 import BundleSerializerV08
from ..bundle.serializer.v09 import BundleSerializerV09
from ..inventorytree import InventoryTree
def test_renames(self):
    """Ensure that file renames have the proper effect on children"""
    btree = self.make_tree_1()[0]
    self.assertEqual(btree.old_path('grandparent'), 'grandparent')
    self.assertEqual(btree.old_path('grandparent/parent'), 'grandparent/parent')
    self.assertEqual(btree.old_path('grandparent/parent/file'), 'grandparent/parent/file')
    self.assertEqual(btree.id2path(b'a'), 'grandparent')
    self.assertEqual(btree.id2path(b'b'), 'grandparent/parent')
    self.assertEqual(btree.id2path(b'c'), 'grandparent/parent/file')
    self.assertEqual(btree.path2id('grandparent'), b'a')
    self.assertEqual(btree.path2id('grandparent/parent'), b'b')
    self.assertEqual(btree.path2id('grandparent/parent/file'), b'c')
    self.assertIs(btree.path2id('grandparent2'), None)
    self.assertIs(btree.path2id('grandparent2/parent'), None)
    self.assertIs(btree.path2id('grandparent2/parent/file'), None)
    btree.note_rename('grandparent', 'grandparent2')
    self.assertIs(btree.old_path('grandparent'), None)
    self.assertIs(btree.old_path('grandparent/parent'), None)
    self.assertIs(btree.old_path('grandparent/parent/file'), None)
    self.assertEqual(btree.id2path(b'a'), 'grandparent2')
    self.assertEqual(btree.id2path(b'b'), 'grandparent2/parent')
    self.assertEqual(btree.id2path(b'c'), 'grandparent2/parent/file')
    self.assertEqual(btree.path2id('grandparent2'), b'a')
    self.assertEqual(btree.path2id('grandparent2/parent'), b'b')
    self.assertEqual(btree.path2id('grandparent2/parent/file'), b'c')
    self.assertTrue(btree.path2id('grandparent') is None)
    self.assertTrue(btree.path2id('grandparent/parent') is None)
    self.assertTrue(btree.path2id('grandparent/parent/file') is None)
    btree.note_rename('grandparent/parent', 'grandparent2/parent2')
    self.assertEqual(btree.id2path(b'a'), 'grandparent2')
    self.assertEqual(btree.id2path(b'b'), 'grandparent2/parent2')
    self.assertEqual(btree.id2path(b'c'), 'grandparent2/parent2/file')
    self.assertEqual(btree.path2id('grandparent2'), b'a')
    self.assertEqual(btree.path2id('grandparent2/parent2'), b'b')
    self.assertEqual(btree.path2id('grandparent2/parent2/file'), b'c')
    self.assertTrue(btree.path2id('grandparent2/parent') is None)
    self.assertTrue(btree.path2id('grandparent2/parent/file') is None)
    btree.note_rename('grandparent/parent/file', 'grandparent2/parent2/file2')
    self.assertEqual(btree.id2path(b'a'), 'grandparent2')
    self.assertEqual(btree.id2path(b'b'), 'grandparent2/parent2')
    self.assertEqual(btree.id2path(b'c'), 'grandparent2/parent2/file2')
    self.assertEqual(btree.path2id('grandparent2'), b'a')
    self.assertEqual(btree.path2id('grandparent2/parent2'), b'b')
    self.assertEqual(btree.path2id('grandparent2/parent2/file2'), b'c')
    self.assertTrue(btree.path2id('grandparent2/parent2/file') is None)