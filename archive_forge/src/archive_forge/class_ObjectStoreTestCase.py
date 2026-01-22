import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
class ObjectStoreTestCase(CompatTestCase):
    """Tests for git repository compatibility."""

    def setUp(self):
        super().setUp()
        self._repo = self.import_repo('server_new.export')

    def _run_git(self, args):
        return run_git_or_fail(args, cwd=self._repo.path)

    def _parse_refs(self, output):
        refs = {}
        for line in BytesIO(output):
            fields = line.rstrip(b'\n').split(b' ')
            self.assertEqual(3, len(fields))
            refname, type_name, sha = fields
            check_ref_format(refname[5:])
            hex_to_sha(sha)
            refs[refname] = (type_name, sha)
        return refs

    def _parse_objects(self, output):
        return {s.rstrip(b'\n').split(b' ')[0] for s in BytesIO(output)}

    def test_bare(self):
        self.assertTrue(self._repo.bare)
        self.assertFalse(os.path.exists(os.path.join(self._repo.path, '.git')))

    def test_head(self):
        output = self._run_git(['rev-parse', 'HEAD'])
        head_sha = output.rstrip(b'\n')
        hex_to_sha(head_sha)
        self.assertEqual(head_sha, self._repo.refs[b'HEAD'])

    def test_refs(self):
        output = self._run_git(['for-each-ref', '--format=%(refname) %(objecttype) %(objectname)'])
        expected_refs = self._parse_refs(output)
        actual_refs = {}
        for refname, sha in self._repo.refs.as_dict().items():
            if refname == b'HEAD':
                continue
            obj = self._repo[sha]
            self.assertEqual(sha, obj.id)
            actual_refs[refname] = (obj.type_name, obj.id)
        self.assertEqual(expected_refs, actual_refs)

    def _get_loose_shas(self):
        output = self._run_git(['rev-list', '--all', '--objects', '--unpacked'])
        return self._parse_objects(output)

    def _get_all_shas(self):
        output = self._run_git(['rev-list', '--all', '--objects'])
        return self._parse_objects(output)

    def assertShasMatch(self, expected_shas, actual_shas_iter):
        actual_shas = set()
        for sha in actual_shas_iter:
            obj = self._repo[sha]
            self.assertEqual(sha, obj.id)
            actual_shas.add(sha)
        self.assertEqual(expected_shas, actual_shas)

    def test_loose_objects(self):
        expected_shas = self._get_loose_shas()
        self.assertShasMatch(expected_shas, self._repo.object_store._iter_loose_objects())

    def test_packed_objects(self):
        expected_shas = self._get_all_shas() - self._get_loose_shas()
        self.assertShasMatch(expected_shas, chain.from_iterable(self._repo.object_store.packs))

    def test_all_objects(self):
        expected_shas = self._get_all_shas()
        self.assertShasMatch(expected_shas, iter(self._repo.object_store))