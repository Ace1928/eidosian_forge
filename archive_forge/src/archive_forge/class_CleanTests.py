import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
class CleanTests(PorcelainTestCase):

    def put_files(self, tracked, ignored, untracked, empty_dirs):
        """Put the described files in the wd."""
        all_files = tracked | ignored | untracked
        for file_path in all_files:
            abs_path = os.path.join(self.repo.path, file_path)
            parent_dir = os.path.dirname(abs_path)
            try:
                os.makedirs(parent_dir)
            except FileExistsError:
                pass
            with open(abs_path, 'w') as f:
                f.write('')
        with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
            f.writelines(ignored)
        for dir_path in empty_dirs:
            os.mkdir(os.path.join(self.repo.path, 'empty_dir'))
        files_to_add = [os.path.join(self.repo.path, t) for t in tracked]
        porcelain.add(repo=self.repo.path, paths=files_to_add)
        porcelain.commit(repo=self.repo.path, message='init commit')

    def assert_wd(self, expected_paths):
        """Assert paths of files and dirs in wd are same as expected_paths."""
        control_dir_rel = os.path.relpath(self.repo._controldir, self.repo.path)
        found_paths = {os.path.normpath(p) for p in flat_walk_dir(self.repo.path) if not p.split(os.sep)[0] == control_dir_rel}
        norm_expected_paths = {os.path.normpath(p) for p in expected_paths}
        self.assertEqual(found_paths, norm_expected_paths)

    def test_from_root(self):
        self.put_files(tracked={'tracked_file', 'tracked_dir/tracked_file', '.gitignore'}, ignored={'ignored_file'}, untracked={'untracked_file', 'tracked_dir/untracked_dir/untracked_file', 'untracked_dir/untracked_dir/untracked_file'}, empty_dirs={'empty_dir'})
        porcelain.clean(repo=self.repo.path, target_dir=self.repo.path)
        self.assert_wd({'tracked_file', 'tracked_dir/tracked_file', '.gitignore', 'ignored_file', 'tracked_dir'})

    def test_from_subdir(self):
        self.put_files(tracked={'tracked_file', 'tracked_dir/tracked_file', '.gitignore'}, ignored={'ignored_file'}, untracked={'untracked_file', 'tracked_dir/untracked_dir/untracked_file', 'untracked_dir/untracked_dir/untracked_file'}, empty_dirs={'empty_dir'})
        porcelain.clean(repo=self.repo, target_dir=os.path.join(self.repo.path, 'untracked_dir'))
        self.assert_wd({'tracked_file', 'tracked_dir/tracked_file', '.gitignore', 'ignored_file', 'untracked_file', 'tracked_dir/untracked_dir/untracked_file', 'empty_dir', 'untracked_dir', 'tracked_dir', 'tracked_dir/untracked_dir'})