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
def test_too_many_names(self):
    fileobj = BytesIO()
    writer = v4.BundleWriter(fileobj)
    writer.begin()
    writer.add_info_record({b'foo': b'bar'})
    writer._container.add_bytes_record([b'blah'], len(b'blah'), [(b'two',), (b'names',)])
    writer.end()
    fileobj.seek(0)
    record_iter = v4.BundleReader(fileobj).iter_records()
    record = next(record_iter)
    self.assertEqual((None, {b'foo': b'bar', b'storage_kind': b'header'}, 'info', None, None), record)
    self.assertRaises(errors.BadBundle, next, record_iter)