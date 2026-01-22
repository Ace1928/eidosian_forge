import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
class WorkingTreeTestCase(ObjectStoreTestCase):
    """Test for compatibility with git-worktree."""
    min_git_version = (2, 5, 0)

    def create_new_worktree(self, repo_dir, branch):
        """Create a new worktree using git-worktree.

        Args:
          repo_dir: The directory of the main working tree.
          branch: The branch or commit to checkout in the new worktree.

        Returns: The path to the new working tree.
        """
        temp_dir = tempfile.mkdtemp()
        run_git_or_fail(['worktree', 'add', temp_dir, branch], cwd=repo_dir)
        self.addCleanup(rmtree_ro, temp_dir)
        return temp_dir

    def setUp(self):
        super().setUp()
        self._worktree_path = self.create_new_worktree(self._repo.path, 'branch')
        self._worktree_repo = Repo(self._worktree_path)
        self.addCleanup(self._worktree_repo.close)
        self._mainworktree_repo = self._repo
        self._number_of_working_tree = 2
        self._repo = self._worktree_repo

    def test_refs(self):
        super().test_refs()
        self.assertEqual(self._mainworktree_repo.refs.allkeys(), self._repo.refs.allkeys())

    def test_head_equality(self):
        self.assertNotEqual(self._repo.refs[b'HEAD'], self._mainworktree_repo.refs[b'HEAD'])

    def test_bare(self):
        self.assertFalse(self._repo.bare)
        self.assertTrue(os.path.isfile(os.path.join(self._repo.path, '.git')))

    def _parse_worktree_list(self, output):
        worktrees = []
        for line in BytesIO(output):
            fields = line.rstrip(b'\n').split()
            worktrees.append(tuple((f.decode() for f in fields)))
        return worktrees

    def test_git_worktree_list(self):
        require_git_version((2, 7, 0))
        output = run_git_or_fail(['worktree', 'list'], cwd=self._repo.path)
        worktrees = self._parse_worktree_list(output)
        self.assertEqual(len(worktrees), self._number_of_working_tree)
        self.assertEqual(worktrees[0][1], '(bare)')
        self.assertTrue(os.path.samefile(worktrees[0][0], self._mainworktree_repo.path))
        output = run_git_or_fail(['worktree', 'list'], cwd=self._mainworktree_repo.path)
        worktrees = self._parse_worktree_list(output)
        self.assertEqual(len(worktrees), self._number_of_working_tree)
        self.assertEqual(worktrees[0][1], '(bare)')
        self.assertTrue(os.path.samefile(worktrees[0][0], self._mainworktree_repo.path))

    def test_git_worktree_config(self):
        """Test that git worktree config parsing matches the git CLI's behavior."""
        require_git_version((2, 7, 0))
        test_name = 'Jelmer'
        test_email = 'jelmer@apache.org'
        run_git_or_fail(['config', 'user.name', test_name], cwd=self._repo.path)
        run_git_or_fail(['config', 'user.email', test_email], cwd=self._repo.path)
        worktree_cfg = self._worktree_repo.get_config()
        main_cfg = self._repo.get_config()
        self.assertEqual(worktree_cfg, main_cfg)
        for c in [worktree_cfg, main_cfg]:
            self.assertEqual(test_name.encode(), c.get((b'user',), b'name'))
            self.assertEqual(test_email.encode(), c.get((b'user',), b'email'))
        output_name = run_git_or_fail(['config', 'user.name'], cwd=self._mainworktree_repo.path).decode().rstrip('\n')
        output_email = run_git_or_fail(['config', 'user.email'], cwd=self._mainworktree_repo.path).decode().rstrip('\n')
        self.assertEqual(test_name, output_name)
        self.assertEqual(test_email, output_email)