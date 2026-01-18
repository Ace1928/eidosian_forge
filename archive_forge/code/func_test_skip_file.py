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
def test_skip_file(self):
    """Make sure we don't accidentally write to the wrong versionedfile"""
    self.tree1 = self.make_branch_and_tree('tree')
    self.b1 = self.tree1.branch
    self.build_tree_contents([('tree/file2', b'contents1')])
    self.tree1.add('file2', ids=b'file2-id')
    self.tree1.commit('rev1', rev_id=b'reva')
    self.build_tree_contents([('tree/file3', b'contents2')])
    self.tree1.add('file3', ids=b'file3-id')
    rev2 = self.tree1.commit('rev2')
    target = self.tree1.controldir.sprout('target').open_workingtree()
    self.build_tree_contents([('tree/file2', b'contents3')])
    self.tree1.commit('rev3', rev_id=b'rev3')
    bundle = self.get_valid_bundle(b'reva', b'rev3')
    if getattr(bundle, 'get_bundle_reader', None) is None:
        raise tests.TestSkipped('Bundle format cannot provide reader')
    file_ids = {(f, r) for b, m, k, r, f in bundle.get_bundle_reader().iter_records() if f is not None}
    self.assertEqual({(b'file2-id', b'rev3'), (b'file3-id', rev2)}, file_ids)
    bundle.install_revisions(target.branch.repository)