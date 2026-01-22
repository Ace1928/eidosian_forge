import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
class GitImportProcessorTests(TestCase):
    """Tests for the GitImportProcessor tests."""

    def setUp(self):
        super().setUp()
        self.repo = MemoryRepo()
        try:
            from ..fastexport import GitImportProcessor
        except ImportError as exc:
            raise SkipTest('python-fastimport not available') from exc
        self.processor = GitImportProcessor(self.repo)

    def test_reset_handler(self):
        from fastimport import commands
        [c1] = build_commit_graph(self.repo.object_store, [[1]])
        cmd = commands.ResetCommand(b'refs/heads/foo', c1.id)
        self.processor.reset_handler(cmd)
        self.assertEqual(c1.id, self.repo.get_refs()[b'refs/heads/foo'])
        self.assertEqual(c1.id, self.processor.last_commit)

    def test_reset_handler_marker(self):
        from fastimport import commands
        [c1, c2] = build_commit_graph(self.repo.object_store, [[1], [2]])
        self.processor.markers[b'10'] = c1.id
        cmd = commands.ResetCommand(b'refs/heads/foo', b':10')
        self.processor.reset_handler(cmd)
        self.assertEqual(c1.id, self.repo.get_refs()[b'refs/heads/foo'])

    def test_reset_handler_default(self):
        from fastimport import commands
        [c1, c2] = build_commit_graph(self.repo.object_store, [[1], [2]])
        cmd = commands.ResetCommand(b'refs/heads/foo', None)
        self.processor.reset_handler(cmd)
        self.assertEqual(ZERO_SHA, self.repo.get_refs()[b'refs/heads/foo'])

    def test_commit_handler(self):
        from fastimport import commands
        cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', None, [], [])
        self.processor.commit_handler(cmd)
        commit = self.repo[self.processor.last_commit]
        self.assertEqual(b'Jelmer <jelmer@samba.org>', commit.author)
        self.assertEqual(b'Jelmer <jelmer@samba.org>', commit.committer)
        self.assertEqual(b'FOO', commit.message)
        self.assertEqual([], commit.parents)
        self.assertEqual(432432432.0, commit.commit_time)
        self.assertEqual(432432432.0, commit.author_time)
        self.assertEqual(3600, commit.commit_timezone)
        self.assertEqual(3600, commit.author_timezone)
        self.assertEqual(commit, self.repo[b'refs/heads/foo'])

    def test_commit_handler_markers(self):
        from fastimport import commands
        [c1, c2, c3] = build_commit_graph(self.repo.object_store, [[1], [2], [3]])
        self.processor.markers[b'10'] = c1.id
        self.processor.markers[b'42'] = c2.id
        self.processor.markers[b'98'] = c3.id
        cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', b':10', [b':42', b':98'], [])
        self.processor.commit_handler(cmd)
        commit = self.repo[self.processor.last_commit]
        self.assertEqual(c1.id, commit.parents[0])
        self.assertEqual(c2.id, commit.parents[1])
        self.assertEqual(c3.id, commit.parents[2])

    def test_import_stream(self):
        markers = self.processor.import_stream(BytesIO(b'blob\nmark :1\ndata 11\ntext for a\n\ncommit refs/heads/master\nmark :2\ncommitter Joe Foo <joe@foo.com> 1288287382 +0000\ndata 20\n<The commit message>\nM 100644 :1 a\n\n'))
        self.assertEqual(2, len(markers))
        self.assertIsInstance(self.repo[markers[b'1']], Blob)
        self.assertIsInstance(self.repo[markers[b'2']], Commit)

    def test_file_add(self):
        from fastimport import commands
        cmd = commands.BlobCommand(b'23', b'data')
        self.processor.blob_handler(cmd)
        cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', None, [], [commands.FileModifyCommand(b'path', 33188, b':23', None)])
        self.processor.commit_handler(cmd)
        commit = self.repo[self.processor.last_commit]
        self.assertEqual([(b'path', 33188, b'6320cd248dd8aeaab759d5871f8781b5c0505172')], self.repo[commit.tree].items())

    def simple_commit(self):
        from fastimport import commands
        cmd = commands.BlobCommand(b'23', b'data')
        self.processor.blob_handler(cmd)
        cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', None, [], [commands.FileModifyCommand(b'path', 33188, b':23', None)])
        self.processor.commit_handler(cmd)
        commit = self.repo[self.processor.last_commit]
        return commit

    def make_file_commit(self, file_cmds):
        """Create a trivial commit with the specified file commands.

        Args:
          file_cmds: File commands to run.
        Returns: The created commit object
        """
        from fastimport import commands
        cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', None, [], file_cmds)
        self.processor.commit_handler(cmd)
        return self.repo[self.processor.last_commit]

    def test_file_copy(self):
        from fastimport import commands
        self.simple_commit()
        commit = self.make_file_commit([commands.FileCopyCommand(b'path', b'new_path')])
        self.assertEqual([(b'new_path', 33188, b'6320cd248dd8aeaab759d5871f8781b5c0505172'), (b'path', 33188, b'6320cd248dd8aeaab759d5871f8781b5c0505172')], self.repo[commit.tree].items())

    def test_file_move(self):
        from fastimport import commands
        self.simple_commit()
        commit = self.make_file_commit([commands.FileRenameCommand(b'path', b'new_path')])
        self.assertEqual([(b'new_path', 33188, b'6320cd248dd8aeaab759d5871f8781b5c0505172')], self.repo[commit.tree].items())

    def test_file_delete(self):
        from fastimport import commands
        self.simple_commit()
        commit = self.make_file_commit([commands.FileDeleteCommand(b'path')])
        self.assertEqual([], self.repo[commit.tree].items())

    def test_file_deleteall(self):
        from fastimport import commands
        self.simple_commit()
        commit = self.make_file_commit([commands.FileDeleteAllCommand()])
        self.assertEqual([], self.repo[commit.tree].items())