import os
import shutil
import stat
import sys
from ...controldir import ControlDir
from .. import KnownFailure, TestCaseWithTransport, TestSkipped
class BisectTestCase(TestCaseWithTransport):
    """Test harness specific to the bisect plugin."""

    def assertRevno(self, rev):
        """Make sure we're at the right revision."""
        rev_contents = {1: 'one', 1.1: 'one dot one', 1.2: 'one dot two', 1.3: 'one dot three', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
        with open('test_file') as test_file:
            content = test_file.read().strip()
        if content != rev_contents[rev]:
            rev_ids = {rev_contents[k]: k for k in rev_contents.keys()}
            found_rev = rev_ids[content]
            raise AssertionError('expected rev %0.1f, found rev %0.1f' % (rev, found_rev))

    def setUp(self):
        """Set up tests."""
        TestCaseWithTransport.setUp(self)
        self.tree = self.make_branch_and_tree('.')
        with open('test_file', 'w') as test_file:
            test_file.write('one')
        self.tree.add(self.tree.relpath(os.path.join(os.getcwd(), 'test_file')))
        with open('test_file_append', 'a') as test_file_append:
            test_file_append.write('one\n')
        self.tree.add(self.tree.relpath(os.path.join(os.getcwd(), 'test_file_append')))
        self.tree.commit(message='add test files')
        ControlDir.open('.').sprout('../temp-clone')
        clone_controldir = ControlDir.open('../temp-clone')
        clone_tree = clone_controldir.open_workingtree()
        for content in ['one dot one', 'one dot two', 'one dot three']:
            with open('../temp-clone/test_file', 'w') as test_file:
                test_file.write(content)
            with open('../temp-clone/test_file_append', 'a') as test_file_append:
                test_file_append.write(content + '\n')
            clone_tree.commit(message='make branch test change')
            saved_subtree_revid = clone_tree.branch.last_revision()
        self.tree.merge_from_branch(clone_tree.branch)
        with open('test_file', 'w') as test_file:
            test_file.write('two')
        with open('test_file_append', 'a') as test_file_append:
            test_file_append.write('two\n')
        self.tree.commit(message='merge external branch')
        shutil.rmtree('../temp-clone')
        self.subtree_rev = saved_subtree_revid
        file_contents = ['three', 'four', 'five']
        for content in file_contents:
            with open('test_file', 'w') as test_file:
                test_file.write(content)
            with open('test_file_append', 'a') as test_file_append:
                test_file_append.write(content + '\n')
            self.tree.commit(message='make test change')

    def testWorkflow(self):
        """Run through a basic usage scenario."""
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.assertRevno(3)
        self.run_bzr(['bisect', 'yes'])
        self.assertRevno(2)
        self.run_bzr(['bisect', 'no'])
        self.assertRevno(3)
        self.run_bzr(['bisect', 'no'])
        self.assertRevno(3)

    def testWorkflowSubtree(self):
        """Run through a usage scenario where the offending change
        is in a subtree."""
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.run_bzr(['bisect', 'yes'])
        self.assertRevno(2)
        self.run_bzr(['bisect', 'yes'])
        self.assertRevno(1.2)
        self.run_bzr(['bisect', 'yes'])
        self.assertRevno(1.1)
        self.run_bzr(['bisect', 'yes'])
        self.assertRevno(1.1)
        self.run_bzr(['bisect', 'yes'])
        self.assertRevno(1.1)

    def testMove(self):
        """Test manually moving to a different revision during the bisection."""
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.run_bzr(['bisect', 'move', '-r', '2'])
        self.assertRevno(2)

    def testReset(self):
        """Test resetting the tree."""
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'reset'])
        self.assertRevno(5)
        with open('test_file', 'w') as test_file:
            test_file.write('keep me')
        out, err = self.run_bzr(['bisect', 'reset'], retcode=3)
        self.assertIn('No bisection in progress.', err)
        with open('test_file') as test_file:
            content = test_file.read().strip()
        self.assertEqual(content, 'keep me')

    def testLog(self):
        """Test saving the current bisection state, and re-loading it."""
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'log', '-o', 'bisect_log'])
        self.run_bzr(['bisect', 'reset'])
        self.run_bzr(['bisect', 'replay', 'bisect_log'])
        self.assertRevno(2)
        self.run_bzr(['bisect', 'no'])
        self.assertRevno(3)

    def testRunScript(self):
        """Make a test script and run it."""
        with open('test_script', 'w') as test_script:
            test_script.write("#!/bin/sh\ngrep -q '^four' test_file_append\n")
        os.chmod('test_script', stat.S_IRWXU)
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.run_bzr(['bisect', 'run', './test_script'])
        self.assertRevno(4)

    def testRunScriptMergePoint(self):
        """Make a test script and run it."""
        if sys.platform == 'win32':
            raise TestSkipped('Unable to run shell script on windows')
        with open('test_script', 'w') as test_script:
            test_script.write("#!/bin/sh\ngrep -q '^two' test_file_append\n")
        os.chmod('test_script', stat.S_IRWXU)
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.run_bzr(['bisect', 'run', './test_script'])
        try:
            self.assertRevno(2)
        except AssertionError:
            raise KnownFailure('bisect does not drill down into merge commits: https://bugs.launchpad.net/bzr-bisect/+bug/539937')

    def testRunScriptSubtree(self):
        """Make a test script and run it."""
        if sys.platform == 'win32':
            raise TestSkipped('Unable to run shell script on windows')
        with open('test_script', 'w') as test_script:
            test_script.write("#!/bin/sh\ngrep -q '^one dot two' test_file_append\n")
        os.chmod('test_script', stat.S_IRWXU)
        self.run_bzr(['bisect', 'start'])
        self.run_bzr(['bisect', 'yes'])
        self.run_bzr(['bisect', 'no', '-r', '1'])
        self.run_bzr(['bisect', 'run', './test_script'])
        try:
            self.assertRevno(1.2)
        except AssertionError:
            raise KnownFailure('bisect does not drill down into merge commits: https://bugs.launchpad.net/bzr-bisect/+bug/539937')