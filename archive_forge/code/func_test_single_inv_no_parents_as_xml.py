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
def test_single_inv_no_parents_as_xml(self):
    self.make_merged_branch()
    sio = self.make_bundle_just_inventories(b'null:', b'a@cset-0-1', [b'a@cset-0-1'])
    reader = v4.BundleReader(sio, stream_input=False)
    records = list(reader.iter_records())
    self.assertEqual(1, len(records))
    bytes, metadata, repo_kind, revision_id, file_id = records[0]
    self.assertIs(None, file_id)
    self.assertEqual(b'a@cset-0-1', revision_id)
    self.assertEqual('inventory', repo_kind)
    self.assertEqual({b'parents': [], b'sha1': b'a13f42b142d544aac9b085c42595d304150e31a2', b'storage_kind': b'mpdiff'}, metadata)
    self.assertEqualDiff(b'i 4\n<inventory format="10" revision_id="a@cset-0-1">\n<directory file_id="root-id" name="" revision="a@cset-0-1" />\n<file file_id="file-id" name="file" parent_id="root-id" revision="a@cset-0-1" text_sha1="09c2f8647e14e49e922b955c194102070597c2d1" text_size="17" />\n</inventory>\n\n', bytes)