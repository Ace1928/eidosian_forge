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
class V4_2aBundleTester(V4BundleTester):

    def bzrdir_format(self):
        return '2a'

    def get_invalid_bundle(self, base_rev_id, rev_id):
        """Create a bundle from base_rev_id -> rev_id in built-in branch.
        Munge the text so that it's invalid.

        :return: The in-memory bundle
        """
        from ..bundle import serializer
        bundle_txt, rev_ids = self.create_bundle_text(base_rev_id, rev_id)
        new_text = self.get_raw(BytesIO(b''.join(bundle_txt)))
        self.assertContainsRe(new_text, b'(?m)B244\n\ni 1\n<inventory')
        new_text = new_text.replace(b'<file file_id="exe-1"', b'<file executable="y" file_id="exe-1"')
        new_text = new_text.replace(b'B244', b'B259')
        bundle_txt = BytesIO()
        bundle_txt.write(serializer._get_bundle_header('4'))
        bundle_txt.write(b'\n')
        bundle_txt.write(bz2.compress(new_text))
        bundle_txt.seek(0)
        bundle = read_bundle(bundle_txt)
        self.valid_apply_bundle(base_rev_id, bundle)
        return bundle

    def make_merged_branch(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'original content\n'))], revision_id=b'a@cset-0-1')
        builder.build_snapshot([b'a@cset-0-1'], [('modify', ('file', b'new-content\n'))], revision_id=b'a@cset-0-2a')
        builder.build_snapshot([b'a@cset-0-1'], [('add', ('other-file', b'file2-id', 'file', b'file2-content\n'))], revision_id=b'a@cset-0-2b')
        builder.build_snapshot([b'a@cset-0-2a', b'a@cset-0-2b'], [('add', ('other-file', b'file2-id', 'file', b'file2-content\n'))], revision_id=b'a@cset-0-3')
        builder.finish_series()
        self.b1 = builder.get_branch()
        self.b1.lock_read()
        self.addCleanup(self.b1.unlock)

    def make_bundle_just_inventories(self, base_revision_id, target_revision_id, revision_ids):
        sio = BytesIO()
        writer = v4.BundleWriteOperation(base_revision_id, target_revision_id, self.b1.repository, sio)
        writer.bundle.begin()
        writer._add_inventory_mpdiffs_from_serializer(revision_ids)
        writer.bundle.end()
        sio.seek(0)
        return sio

    def test_single_inventory_multiple_parents_as_xml(self):
        self.make_merged_branch()
        sio = self.make_bundle_just_inventories(b'a@cset-0-1', b'a@cset-0-3', [b'a@cset-0-3'])
        reader = v4.BundleReader(sio, stream_input=False)
        records = list(reader.iter_records())
        self.assertEqual(1, len(records))
        bytes, metadata, repo_kind, revision_id, file_id = records[0]
        self.assertIs(None, file_id)
        self.assertEqual(b'a@cset-0-3', revision_id)
        self.assertEqual('inventory', repo_kind)
        self.assertEqual({b'parents': [b'a@cset-0-2a', b'a@cset-0-2b'], b'sha1': b'09c53b0c4de0895e11a2aacc34fef60a6e70865c', b'storage_kind': b'mpdiff'}, metadata)
        self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-3">\n\nc 0 1 1 2\nc 1 3 3 2\n', bytes)

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

    def test_multiple_inventories_as_xml(self):
        self.make_merged_branch()
        sio = self.make_bundle_just_inventories(b'a@cset-0-1', b'a@cset-0-3', [b'a@cset-0-2a', b'a@cset-0-2b', b'a@cset-0-3'])
        reader = v4.BundleReader(sio, stream_input=False)
        records = list(reader.iter_records())
        self.assertEqual(3, len(records))
        revision_ids = [rev_id for b, m, k, rev_id, f in records]
        self.assertEqual([b'a@cset-0-2a', b'a@cset-0-2b', b'a@cset-0-3'], revision_ids)
        metadata_2a = records[0][1]
        self.assertEqual({b'parents': [b'a@cset-0-1'], b'sha1': b'1e105886d62d510763e22885eec733b66f5f09bf', b'storage_kind': b'mpdiff'}, metadata_2a)
        metadata_2b = records[1][1]
        self.assertEqual({b'parents': [b'a@cset-0-1'], b'sha1': b'f03f12574bdb5ed2204c28636c98a8547544ccd8', b'storage_kind': b'mpdiff'}, metadata_2b)
        metadata_3 = records[2][1]
        self.assertEqual({b'parents': [b'a@cset-0-2a', b'a@cset-0-2b'], b'sha1': b'09c53b0c4de0895e11a2aacc34fef60a6e70865c', b'storage_kind': b'mpdiff'}, metadata_3)
        bytes_2a = records[0][0]
        self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-2a">\n\nc 0 1 1 1\ni 1\n<file file_id="file-id" name="file" parent_id="root-id" revision="a@cset-0-2a" text_sha1="50f545ff40e57b6924b1f3174b267ffc4576e9a9" text_size="12" />\n\nc 0 3 3 1\n', bytes_2a)
        bytes_2b = records[1][0]
        self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-2b">\n\nc 0 1 1 2\ni 1\n<file file_id="file2-id" name="other-file" parent_id="root-id" revision="a@cset-0-2b" text_sha1="b46c0c8ea1e5ef8e46fc8894bfd4752a88ec939e" text_size="14" />\n\nc 0 3 4 1\n', bytes_2b)
        bytes_3 = records[2][0]
        self.assertEqualDiff(b'i 1\n<inventory format="10" revision_id="a@cset-0-3">\n\nc 0 1 1 2\nc 1 3 3 2\n', bytes_3)

    def test_creating_bundle_preserves_chk_pages(self):
        self.make_merged_branch()
        target = self.b1.controldir.sprout('target', revision_id=b'a@cset-0-2a').open_branch()
        bundle_txt, rev_ids = self.create_bundle_text(b'a@cset-0-2a', b'a@cset-0-3')
        self.assertEqual({b'a@cset-0-2b', b'a@cset-0-3'}, set(rev_ids))
        bundle = read_bundle(bundle_txt)
        target.lock_write()
        self.addCleanup(target.unlock)
        install_bundle(target.repository, bundle)
        inv1 = next(self.b1.repository.inventories.get_record_stream([(b'a@cset-0-3',)], 'unordered', True)).get_bytes_as('fulltext')
        inv2 = next(target.repository.inventories.get_record_stream([(b'a@cset-0-3',)], 'unordered', True)).get_bytes_as('fulltext')
        self.assertEqualDiff(inv1, inv2)