from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class TestWithBrokenRepo(TestCaseWithTransport):
    """These tests seem to be more appropriate as interface tests?"""

    def make_broken_repository(self):
        repo = self.make_repository('broken-repo')
        cleanups = []
        try:
            repo.lock_write()
            cleanups.append(repo.unlock)
            repo.start_write_group()
            cleanups.append(repo.commit_write_group)
            inv = inventory.Inventory(revision_id=b'rev1a')
            inv.root.revision = b'rev1a'
            self.add_file(repo, inv, 'file1', b'rev1a', [])
            repo.texts.add_lines((inv.root.file_id, b'rev1a'), [], [])
            repo.add_inventory(b'rev1a', inv, [])
            revision = _mod_revision.Revision(b'rev1a', committer='jrandom@example.com', timestamp=0, inventory_sha1='', timezone=0, message='foo', parent_ids=[])
            repo.add_revision(b'rev1a', revision, inv)
            inv = inventory.Inventory(revision_id=b'rev1b')
            inv.root.revision = b'rev1b'
            self.add_file(repo, inv, 'file1', b'rev1b', [])
            repo.add_inventory(b'rev1b', inv, [])
            inv = inventory.Inventory()
            self.add_file(repo, inv, 'file1', b'rev2', [b'rev1a', b'rev1b'])
            self.add_file(repo, inv, 'file2', b'rev2', [])
            self.add_revision(repo, b'rev2', inv, [b'rev1a'])
            inv = inventory.Inventory()
            self.add_file(repo, inv, 'file2', b'rev1c', [])
            inv = inventory.Inventory()
            self.add_file(repo, inv, 'file2', b'rev3', [b'rev1c'])
            self.add_revision(repo, b'rev3', inv, [b'rev1c'])
            return repo
        finally:
            for cleanup in reversed(cleanups):
                cleanup()

    def add_revision(self, repo, revision_id, inv, parent_ids):
        inv.revision_id = revision_id
        inv.root.revision = revision_id
        repo.texts.add_lines((inv.root.file_id, revision_id), [], [])
        repo.add_inventory(revision_id, inv, parent_ids)
        revision = _mod_revision.Revision(revision_id, committer='jrandom@example.com', timestamp=0, inventory_sha1='', timezone=0, message='foo', parent_ids=parent_ids)
        repo.add_revision(revision_id, revision, inv)

    def add_file(self, repo, inv, filename, revision, parents):
        file_id = filename.encode('utf-8') + b'-id'
        content = [b'line\n']
        entry = inventory.InventoryFile(file_id, filename, b'TREE_ROOT')
        entry.revision = revision
        entry.text_sha1 = osutils.sha_strings(content)
        entry.text_size = 0
        inv.add(entry)
        text_key = (file_id, revision)
        parent_keys = [(file_id, parent) for parent in parents]
        repo.texts.add_lines(text_key, parent_keys, content)

    def test_insert_from_broken_repo(self):
        """Inserting a data stream from a broken repository won't silently
        corrupt the target repository.
        """
        broken_repo = self.make_broken_repository()
        empty_repo = self.make_repository('empty-repo')
        try:
            empty_repo.fetch(broken_repo)
        except (errors.RevisionNotPresent, errors.BzrCheckError):
            return
        empty_repo.lock_read()
        self.addCleanup(empty_repo.unlock)
        text = next(empty_repo.texts.get_record_stream([(b'file2-id', b'rev3')], 'topological', True))
        self.assertEqual(b'line\n', text.get_bytes_as('fulltext'))