import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
class BranchStatus(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        status.hooks.install_named_hook('post_status', status._show_shelve_summary, 'brz status')

    def assertStatus(self, expected_lines, working_tree, specific_files=None, revision=None, short=False, pending=True, verbose=False):
        """Run status in working_tree and look for output.

        :param expected_lines: The lines to look for.
        :param working_tree: The tree to run status in.
        """
        output_string = self.status_string(working_tree, specific_files, revision, short, pending, verbose)
        self.assertEqual(expected_lines, output_string.splitlines(True))

    def status_string(self, wt, specific_files=None, revision=None, short=False, pending=True, verbose=False):
        uio = self.make_utf8_encoded_stringio()
        show_tree_status(wt, specific_files=specific_files, to_file=uio, revision=revision, short=short, show_pending=pending, verbose=verbose)
        return uio.getvalue().decode('utf-8')

    def test_branch_status(self):
        """Test basic branch status"""
        wt = self.make_branch_and_tree('.')
        self.assertStatus([], wt)
        self.build_tree(['hello.c', 'bye.c'])
        self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n'], wt)
        self.assertStatus(['?   bye.c\n', '?   hello.c\n'], wt, short=True)
        wt.commit('create a parent to allow testing merge output')
        wt.add_parent_tree_id(b'pending@pending-0-0')
        self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n', 'pending merge tips: (use -v to see all merge revisions)\n', '  (ghost) pending@pending-0-0\n'], wt)
        self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n', 'pending merges:\n', '  (ghost) pending@pending-0-0\n'], wt, verbose=True)
        self.assertStatus(['?   bye.c\n', '?   hello.c\n', 'P   (ghost) pending@pending-0-0\n'], wt, short=True)
        self.assertStatus(['unknown:\n', '  bye.c\n', '  hello.c\n'], wt, pending=False)
        self.assertStatus(['?   bye.c\n', '?   hello.c\n'], wt, short=True, pending=False)

    def test_branch_status_revisions(self):
        """Tests branch status with revisions"""
        wt = self.make_branch_and_tree('.')
        self.build_tree(['hello.c', 'bye.c'])
        wt.add('hello.c')
        wt.add('bye.c')
        wt.commit('Test message')
        revs = [RevisionSpec.from_string('0')]
        self.assertStatus(['added:\n', '  bye.c\n', '  hello.c\n'], wt, revision=revs)
        self.build_tree(['more.c'])
        wt.add('more.c')
        wt.commit('Another test message')
        revs.append(RevisionSpec.from_string('1'))
        self.assertStatus(['added:\n', '  bye.c\n', '  hello.c\n'], wt, revision=revs)

    def test_pending(self):
        """Pending merges display works, including Unicode"""
        mkdir('./branch')
        wt = self.make_branch_and_tree('branch')
        b = wt.branch
        wt.commit('Empty commit 1')
        b_2_dir = b.controldir.sprout('./copy')
        b_2 = b_2_dir.open_branch()
        wt2 = b_2_dir.open_workingtree()
        wt.commit('༢ Empty commit 2')
        wt2.merge_from_branch(wt.branch)
        message = self.status_string(wt2, verbose=True)
        self.assertStartsWith(message, 'pending merges:\n')
        self.assertEndsWith(message, 'Empty commit 2\n')
        wt2.commit('merged')
        wt.commit('Empty commit 3 ' + 'blah blah blah blah ' * 100)
        wt2.merge_from_branch(wt.branch)
        message = self.status_string(wt2, verbose=True)
        self.assertStartsWith(message, 'pending merges:\n')
        self.assertTrue('Empty commit 3' in message)
        self.assertEndsWith(message, '...\n')

    def test_tree_status_ignores(self):
        """Tests branch status with ignores"""
        wt = self.make_branch_and_tree('.')
        self.run_bzr('ignore *~')
        wt.commit('commit .bzrignore')
        self.build_tree(['foo.c', 'foo.c~'])
        self.assertStatus(['unknown:\n', '  foo.c\n'], wt)
        self.assertStatus(['?   foo.c\n'], wt, short=True)

    def test_tree_status_specific_files(self):
        """Tests branch status with given specific files"""
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        self.build_tree(['directory/', 'directory/hello.c', 'bye.c', 'test.c', 'dir2/', 'missing.c'])
        wt.add('directory')
        wt.add('test.c')
        wt.commit('testing')
        wt.add('missing.c')
        unlink('missing.c')
        self.assertStatus(['missing:\n', '  missing.c\n', 'unknown:\n', '  bye.c\n', '  dir2/\n', '  directory/hello.c\n'], wt)
        self.assertStatus(['?   bye.c\n', '?   dir2/\n', '?   directory/hello.c\n', '+!  missing.c\n'], wt, short=True)
        tof = StringIO()
        self.assertRaises(errors.PathsDoNotExist, show_tree_status, wt, specific_files=['bye.c', 'test.c', 'absent.c'], to_file=tof)
        tof = StringIO()
        show_tree_status(wt, specific_files=['directory'], to_file=tof)
        tof.seek(0)
        self.assertEqual(tof.readlines(), ['unknown:\n', '  directory/hello.c\n'])
        tof = StringIO()
        show_tree_status(wt, specific_files=['directory'], to_file=tof, short=True)
        tof.seek(0)
        self.assertEqual(tof.readlines(), ['?   directory/hello.c\n'])
        tof = StringIO()
        show_tree_status(wt, specific_files=['dir2'], to_file=tof)
        tof.seek(0)
        self.assertEqual(tof.readlines(), ['unknown:\n', '  dir2/\n'])
        tof = StringIO()
        show_tree_status(wt, specific_files=['dir2'], to_file=tof, short=True)
        tof.seek(0)
        self.assertEqual(tof.readlines(), ['?   dir2/\n'])
        tof = StringIO()
        revs = [RevisionSpec.from_string('0'), RevisionSpec.from_string('1')]
        show_tree_status(wt, specific_files=['test.c'], to_file=tof, short=True, revision=revs)
        tof.seek(0)
        self.assertEqual(tof.readlines(), ['+N  test.c\n'])
        tof = StringIO()
        show_tree_status(wt, specific_files=['missing.c'], to_file=tof)
        tof.seek(0)
        self.assertEqual(tof.readlines(), ['missing:\n', '  missing.c\n'])
        tof = StringIO()
        show_tree_status(wt, specific_files=['missing.c'], to_file=tof, short=True)
        tof.seek(0)
        self.assertEqual(tof.readlines(), ['+!  missing.c\n'])

    def test_specific_files_conflicts(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['dir2/'])
        tree.add('dir2')
        tree.commit('added dir2')
        tree.set_conflicts([conflicts.ContentsConflict('foo')])
        tof = BytesIO()
        show_tree_status(tree, specific_files=['dir2'], to_file=tof)
        self.assertEqualDiff(b'', tof.getvalue())
        tree.set_conflicts([conflicts.ContentsConflict('dir2')])
        tof = StringIO()
        show_tree_status(tree, specific_files=['dir2'], to_file=tof)
        self.assertEqualDiff('conflicts:\n  Contents conflict in dir2\n', tof.getvalue())
        tree.set_conflicts([conflicts.ContentsConflict('dir2/file1')])
        tof = StringIO()
        show_tree_status(tree, specific_files=['dir2'], to_file=tof)
        self.assertEqualDiff('conflicts:\n  Contents conflict in dir2/file1\n', tof.getvalue())

    def _prepare_nonexistent(self):
        wt = self.make_branch_and_tree('.')
        self.assertStatus([], wt)
        self.build_tree(['FILE_A', 'FILE_B', 'FILE_C', 'FILE_D', 'FILE_E'])
        wt.add('FILE_A')
        wt.add('FILE_B')
        wt.add('FILE_C')
        wt.add('FILE_D')
        wt.add('FILE_E')
        wt.commit('Create five empty files.')
        with open('FILE_B', 'w') as f:
            f.write('Modification to file FILE_B.')
        with open('FILE_C', 'w') as f:
            f.write('Modification to file FILE_C.')
        unlink('FILE_E')
        with open('FILE_Q', 'w') as f:
            f.write('FILE_Q is added but not committed.')
        wt.add('FILE_Q')
        open('UNVERSIONED_BUT_EXISTING', 'w')
        return wt

    def test_status_nonexistent_file(self):
        wt = self._prepare_nonexistent()
        self.assertStatus(['removed:\n', '  FILE_E\n', 'added:\n', '  FILE_Q\n', 'modified:\n', '  FILE_B\n', '  FILE_C\n', 'unknown:\n', '  UNVERSIONED_BUT_EXISTING\n'], wt)
        self.assertStatus([' M  FILE_B\n', ' M  FILE_C\n', ' D  FILE_E\n', '+N  FILE_Q\n', '?   UNVERSIONED_BUT_EXISTING\n'], wt, short=True)
        expected = ['nonexistent:\n', '  NONEXISTENT\n']
        out, err = self.run_bzr('status NONEXISTENT', retcode=3)
        self.assertEqual(expected, out.splitlines(True))
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')
        expected = ['X:   NONEXISTENT\n']
        out, err = self.run_bzr('status --short NONEXISTENT', retcode=3)
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')

    def test_status_nonexistent_file_with_others(self):
        wt = self._prepare_nonexistent()
        expected = ['removed:\n', '  FILE_E\n', 'modified:\n', '  FILE_B\n', '  FILE_C\n', 'nonexistent:\n', '  NONEXISTENT\n']
        out, err = self.run_bzr('status NONEXISTENT FILE_A FILE_B FILE_C FILE_D FILE_E', retcode=3)
        self.assertEqual(expected, out.splitlines(True))
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')
        expected = [' M  FILE_B\n', ' M  FILE_C\n', ' D  FILE_E\n', 'X   NONEXISTENT\n']
        out, err = self.run_bzr('status --short NONEXISTENT FILE_A FILE_B FILE_C FILE_D FILE_E', retcode=3)
        self.assertEqual(expected, out.splitlines(True))
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')

    def test_status_multiple_nonexistent_files(self):
        wt = self._prepare_nonexistent()
        expected = ['removed:\n', '  FILE_E\n', 'modified:\n', '  FILE_B\n', '  FILE_C\n', 'nonexistent:\n', '  ANOTHER_NONEXISTENT\n', '  NONEXISTENT\n']
        out, err = self.run_bzr('status NONEXISTENT FILE_A FILE_B ANOTHER_NONEXISTENT FILE_C FILE_D FILE_E', retcode=3)
        self.assertEqual(expected, out.splitlines(True))
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: ANOTHER_NONEXISTENT NONEXISTENT.*')
        expected = [' M  FILE_B\n', ' M  FILE_C\n', ' D  FILE_E\n', 'X   ANOTHER_NONEXISTENT\n', 'X   NONEXISTENT\n']
        out, err = self.run_bzr('status --short NONEXISTENT FILE_A FILE_B ANOTHER_NONEXISTENT FILE_C FILE_D FILE_E', retcode=3)
        self.assertEqual(expected, out.splitlines(True))
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: ANOTHER_NONEXISTENT NONEXISTENT.*')

    def test_status_nonexistent_file_with_unversioned(self):
        wt = self._prepare_nonexistent()
        expected = ['removed:\n', '  FILE_E\n', 'added:\n', '  FILE_Q\n', 'modified:\n', '  FILE_B\n', '  FILE_C\n', 'unknown:\n', '  UNVERSIONED_BUT_EXISTING\n', 'nonexistent:\n', '  NONEXISTENT\n']
        out, err = self.run_bzr('status NONEXISTENT FILE_A FILE_B UNVERSIONED_BUT_EXISTING FILE_C FILE_D FILE_E FILE_Q', retcode=3)
        self.assertEqual(expected, out.splitlines(True))
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')
        expected = sorted(['+N  FILE_Q\n', '?   UNVERSIONED_BUT_EXISTING\n', ' D  FILE_E\n', ' M  FILE_C\n', ' M  FILE_B\n', 'X   NONEXISTENT\n'])
        out, err = self.run_bzr('status --short NONEXISTENT FILE_A FILE_B UNVERSIONED_BUT_EXISTING FILE_C FILE_D FILE_E FILE_Q', retcode=3)
        actual = out.splitlines(True)
        actual.sort()
        self.assertEqual(expected, actual)
        self.assertContainsRe(err, '.*ERROR: Path\\(s\\) do not exist: NONEXISTENT.*')

    def test_status_out_of_date(self):
        """Simulate status of out-of-date tree after remote push"""
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', b'foo\n')])
        with tree.lock_write():
            tree.add(['a'])
            tree.commit('add test file')
            tree.set_last_revision(b'0')
        out, err = self.run_bzr('status')
        self.assertEqual("working tree is out of date, run 'brz update'\n", err)

    def test_status_on_ignored(self):
        """Tests branch status on an unversioned file which is considered ignored.

        See https://bugs.launchpad.net/bzr/+bug/40103
        """
        tree = self.make_branch_and_tree('.')
        self.build_tree(['test1.c', 'test1.c~', 'test2.c~'])
        result = self.run_bzr('status')[0]
        self.assertContainsRe(result, 'unknown:\n  test1.c\n')
        short_result = self.run_bzr('status --short')[0]
        self.assertContainsRe(short_result, '\\?   test1.c\n')
        result = self.run_bzr('status test1.c')[0]
        self.assertContainsRe(result, 'unknown:\n  test1.c\n')
        short_result = self.run_bzr('status --short test1.c')[0]
        self.assertContainsRe(short_result, '\\?   test1.c\n')
        result = self.run_bzr('status test1.c~')[0]
        self.assertContainsRe(result, 'ignored:\n  test1.c~\n')
        short_result = self.run_bzr('status --short test1.c~')[0]
        self.assertContainsRe(short_result, 'I   test1.c~\n')
        result = self.run_bzr('status test1.c~ test2.c~')[0]
        self.assertContainsRe(result, 'ignored:\n  test1.c~\n  test2.c~\n')
        short_result = self.run_bzr('status --short test1.c~ test2.c~')[0]
        self.assertContainsRe(short_result, 'I   test1.c~\nI   test2.c~\n')
        result = self.run_bzr('status test1.c test1.c~ test2.c~')[0]
        self.assertContainsRe(result, 'unknown:\n  test1.c\nignored:\n  test1.c~\n  test2.c~\n')
        short_result = self.run_bzr('status --short test1.c test1.c~ test2.c~')[0]
        self.assertContainsRe(short_result, '\\?   test1.c\nI   test1.c~\nI   test2.c~\n')

    def test_status_write_lock(self):
        """Test that status works without fetching history and
        having a write lock.

        See https://bugs.launchpad.net/bzr/+bug/149270
        """
        mkdir('branch1')
        wt = self.make_branch_and_tree('branch1')
        b = wt.branch
        wt.commit('Empty commit 1')
        wt2 = b.controldir.sprout('branch2').open_workingtree()
        wt2.commit('Empty commit 2')
        out, err = self.run_bzr('status branch1 -rbranch:branch2')
        self.assertEqual('', out)

    def test_status_with_shelves(self):
        """Ensure that _show_shelve_summary handler works.
        """
        wt = self.make_branch_and_tree('.')
        self.build_tree(['hello.c'])
        wt.add('hello.c')
        self.run_bzr(['shelve', '--all', '-m', 'foo'])
        self.build_tree(['bye.c'])
        wt.add('bye.c')
        self.assertStatus(['added:\n', '  bye.c\n', '1 shelf exists. See "brz shelve --list" for details.\n'], wt)
        self.run_bzr(['shelve', '--all', '-m', 'bar'])
        self.build_tree(['eggs.c', 'spam.c'])
        wt.add('eggs.c')
        wt.add('spam.c')
        self.assertStatus(['added:\n', '  eggs.c\n', '  spam.c\n', '2 shelves exist. See "brz shelve --list" for details.\n'], wt)
        self.assertStatus(['added:\n', '  spam.c\n'], wt, specific_files=['spam.c'])