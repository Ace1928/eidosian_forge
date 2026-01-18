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
def test_whitespace_bundle(self):
    if sys.platform in ('win32', 'cygwin'):
        raise tests.TestSkipped("Windows doesn't support filenames with tabs or trailing spaces")
    self.tree1 = self.make_branch_and_tree('b1')
    self.b1 = self.tree1.branch
    self.build_tree(['b1/trailing space '])
    self.tree1.add(['trailing space '])
    self.tree1.commit('funky whitespace', rev_id=b'white-1')
    bundle = self.get_valid_bundle(b'null:', b'white-1')
    with open('b1/trailing space ', 'ab') as f:
        f.write(b'add some text\n')
    self.tree1.commit('add text', rev_id=b'white-2')
    bundle = self.get_valid_bundle(b'white-1', b'white-2')
    self.tree1.rename_one('trailing space ', ' start and end space ')
    self.tree1.commit('rename', rev_id=b'white-3')
    bundle = self.get_valid_bundle(b'white-2', b'white-3')
    self.tree1.remove([' start and end space '])
    self.tree1.commit('removed', rev_id=b'white-4')
    bundle = self.get_valid_bundle(b'white-3', b'white-4')
    bundle = self.get_valid_bundle(b'null:', b'white-4')