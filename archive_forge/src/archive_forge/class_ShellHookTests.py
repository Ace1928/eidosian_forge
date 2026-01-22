import os
import shutil
import stat
import sys
import tempfile
from dulwich import errors
from dulwich.tests import TestCase
from ..hooks import CommitMsgShellHook, PostCommitShellHook, PreCommitShellHook
class ShellHookTests(TestCase):

    def setUp(self):
        super().setUp()
        if os.name != 'posix':
            self.skipTest('shell hook tests requires POSIX shell')
        self.assertTrue(os.path.exists('/bin/sh'))

    def test_hook_pre_commit(self):
        repo_dir = os.path.join(tempfile.mkdtemp())
        os.mkdir(os.path.join(repo_dir, 'hooks'))
        self.addCleanup(shutil.rmtree, repo_dir)
        pre_commit_fail = '#!/bin/sh\nexit 1\n'
        pre_commit_success = '#!/bin/sh\nexit 0\n'
        pre_commit_cwd = '#!/bin/sh\nif [ "$(pwd)" != \'' + repo_dir + '\' ]; then\n    echo "Expected path \'' + repo_dir + '\', got \'$(pwd)\'"\n    exit 1\nfi\n\nexit 0\n'
        pre_commit = os.path.join(repo_dir, 'hooks', 'pre-commit')
        hook = PreCommitShellHook(repo_dir, repo_dir)
        with open(pre_commit, 'w') as f:
            f.write(pre_commit_fail)
        os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        self.assertRaises(errors.HookError, hook.execute)
        if sys.platform != 'darwin':
            with open(pre_commit, 'w') as f:
                f.write(pre_commit_cwd)
            os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
            hook.execute()
        with open(pre_commit, 'w') as f:
            f.write(pre_commit_success)
        os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        hook.execute()

    def test_hook_commit_msg(self):
        repo_dir = os.path.join(tempfile.mkdtemp())
        os.mkdir(os.path.join(repo_dir, 'hooks'))
        self.addCleanup(shutil.rmtree, repo_dir)
        commit_msg_fail = '#!/bin/sh\nexit 1\n'
        commit_msg_success = '#!/bin/sh\nexit 0\n'
        commit_msg_cwd = '#!/bin/sh\nif [ "$(pwd)" = \'' + repo_dir + "' ]; then exit 0; else exit 1; fi\n"
        commit_msg = os.path.join(repo_dir, 'hooks', 'commit-msg')
        hook = CommitMsgShellHook(repo_dir)
        with open(commit_msg, 'w') as f:
            f.write(commit_msg_fail)
        os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        self.assertRaises(errors.HookError, hook.execute, b'failed commit')
        if sys.platform != 'darwin':
            with open(commit_msg, 'w') as f:
                f.write(commit_msg_cwd)
            os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
            hook.execute(b'cwd test commit')
        with open(commit_msg, 'w') as f:
            f.write(commit_msg_success)
        os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        hook.execute(b'empty commit')

    def test_hook_post_commit(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        repo_dir = os.path.join(tempfile.mkdtemp())
        os.mkdir(os.path.join(repo_dir, 'hooks'))
        self.addCleanup(shutil.rmtree, repo_dir)
        post_commit_success = '#!/bin/sh\nrm ' + path + '\n'
        post_commit_fail = '#!/bin/sh\nexit 1\n'
        post_commit_cwd = '#!/bin/sh\nif [ "$(pwd)" = \'' + repo_dir + "' ]; then exit 0; else exit 1; fi\n"
        post_commit = os.path.join(repo_dir, 'hooks', 'post-commit')
        hook = PostCommitShellHook(repo_dir)
        with open(post_commit, 'w') as f:
            f.write(post_commit_fail)
        os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        self.assertRaises(errors.HookError, hook.execute)
        if sys.platform != 'darwin':
            with open(post_commit, 'w') as f:
                f.write(post_commit_cwd)
            os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
            hook.execute()
        with open(post_commit, 'w') as f:
            f.write(post_commit_success)
        os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        hook.execute()
        self.assertFalse(os.path.exists(path))