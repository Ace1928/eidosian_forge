import errno
import functools
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Tuple
from dulwich.tests import SkipTest, TestCase
from ...protocol import TCP_GIT_PORT
from ...repo import Repo
class CompatTestCase(TestCase):
    """Test case that requires git for compatibility checks.

    Subclasses can change the git version required by overriding
    min_git_version.
    """
    min_git_version: Tuple[int, ...] = (1, 5, 0)

    def setUp(self):
        super().setUp()
        require_git_version(self.min_git_version)

    def assertObjectStoreEqual(self, store1, store2):
        self.assertEqual(sorted(set(store1)), sorted(set(store2)))

    def assertReposEqual(self, repo1, repo2):
        self.assertEqual(repo1.get_refs(), repo2.get_refs())
        self.assertObjectStoreEqual(repo1.object_store, repo2.object_store)

    def assertReposNotEqual(self, repo1, repo2):
        refs1 = repo1.get_refs()
        objs1 = set(repo1.object_store)
        refs2 = repo2.get_refs()
        objs2 = set(repo2.object_store)
        self.assertFalse(refs1 == refs2 and objs1 == objs2)

    def import_repo(self, name):
        """Import a repo from a fast-export file in a temporary directory.

        Args:
          name: The name of the repository export file, relative to
            dulwich/tests/data/repos.
        Returns: An initialized Repo object that lives in a temporary
            directory.
        """
        path = import_repo_to_dir(name)
        repo = Repo(path)

        def cleanup():
            repo.close()
            rmtree_ro(os.path.dirname(path.rstrip(os.sep)))
        self.addCleanup(cleanup)
        return repo