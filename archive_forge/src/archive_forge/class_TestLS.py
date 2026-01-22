from breezy import ignores, tests
class TestLS(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        ignores._set_user_ignores(['user-ignore'])
        self.wt = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', b'*.pyo\n'), ('a', b'hello\n')])

    def ls_equals(self, value, args=None, recursive=True, working_dir=None):
        command = 'ls'
        if args is not None:
            command += ' ' + args
        if recursive:
            command += ' -R'
        out, err = self.run_bzr(command, working_dir=working_dir)
        self.assertEqual('', err)
        self.assertEqualDiff(value, out)

    def test_ls_null_verbose(self):
        self.run_bzr_error(['Cannot set both --verbose and --null'], 'ls --verbose --null')

    def test_ls_basic(self):
        """Test the abilities of 'brz ls'"""
        self.ls_equals('.bzrignore\na\n')
        self.ls_equals('.bzrignore\na\n', './')
        self.ls_equals('?        .bzrignore\n?        a\n', '--verbose')
        self.ls_equals('.bzrignore\na\n', '--unknown')
        self.ls_equals('', '--ignored')
        self.ls_equals('', '--versioned')
        self.ls_equals('', '-V')
        self.ls_equals('.bzrignore\na\n', '--unknown --ignored --versioned')
        self.ls_equals('.bzrignore\na\n', '--unknown --ignored -V')
        self.ls_equals('', '--ignored --versioned')
        self.ls_equals('', '--ignored -V')
        self.ls_equals('.bzrignore\x00a\x00', '--null')

    def test_ls_added(self):
        self.wt.add(['a'])
        self.ls_equals('?        .bzrignore\nV        a\n', '--verbose')
        self.wt.commit('add')
        self.build_tree(['subdir/'])
        self.ls_equals('?        .bzrignore\nV        a\n?        subdir/\n', '--verbose')
        self.build_tree(['subdir/b'])
        self.wt.add(['subdir/', 'subdir/b', '.bzrignore'])
        self.ls_equals('V        .bzrignore\nV        a\nV        subdir/\nV        subdir/b\n', '--verbose')

    def test_show_ids(self):
        self.build_tree(['subdir/'])
        self.wt.add(['a', 'subdir'], ids=[b'a-id', b'subdir-id'])
        self.ls_equals('.bzrignore                                         \na                                                  a-id\nsubdir/                                            subdir-id\n', '--show-ids')
        self.ls_equals('?        .bzrignore\nV        a                                         a-id\nV        subdir/                                   subdir-id\n', '--show-ids --verbose')
        self.ls_equals('.bzrignore\x00\x00a\x00a-id\x00subdir\x00subdir-id\x00', '--show-ids --null')

    def test_ls_no_recursive(self):
        self.build_tree(['subdir/', 'subdir/b'])
        self.wt.add(['a', 'subdir/', 'subdir/b', '.bzrignore'])
        self.ls_equals('.bzrignore\na\nsubdir/\n', recursive=False)
        self.ls_equals('V        .bzrignore\nV        a\nV        subdir/\n', '--verbose', recursive=False)
        self.ls_equals('b\n', working_dir='subdir')
        self.ls_equals('b\x00', '--null', working_dir='subdir')
        self.ls_equals('subdir/b\n', '--from-root', working_dir='subdir')
        self.ls_equals('subdir/b\x00', '--from-root --null', working_dir='subdir')
        self.ls_equals('subdir/b\n', '--from-root', recursive=False, working_dir='subdir')

    def test_ls_path(self):
        """If a path is specified, files are listed with that prefix"""
        self.build_tree(['subdir/', 'subdir/b'])
        self.wt.add(['subdir', 'subdir/b'])
        self.ls_equals('subdir/b\n', 'subdir')
        self.ls_equals('../.bzrignore\n../a\n../subdir/\n../subdir/b\n', '..', working_dir='subdir')
        self.ls_equals('../.bzrignore\x00../a\x00../subdir\x00../subdir/b\x00', '.. --null', working_dir='subdir')
        self.ls_equals('?        ../.bzrignore\n?        ../a\nV        ../subdir/\nV        ../subdir/b\n', '.. --verbose', working_dir='subdir')
        self.run_bzr_error(['cannot specify both --from-root and PATH'], 'ls --from-root ..', working_dir='subdir')

    def test_ls_revision(self):
        self.wt.add(['a'])
        self.wt.commit('add')
        self.build_tree(['subdir/'])
        self.ls_equals('a\n', '--revision 1')
        self.ls_equals('V        a\n', '--verbose --revision 1')
        self.ls_equals('', '--revision 1', working_dir='subdir')

    def test_ls_branch(self):
        """If a branch is specified, files are listed from it"""
        self.build_tree(['subdir/', 'subdir/b'])
        self.wt.add(['subdir', 'subdir/b'])
        self.wt.commit('committing')
        branch = self.make_branch('branchdir')
        branch.pull(self.wt.branch)
        self.ls_equals('branchdir/subdir/\nbranchdir/subdir/b\n', 'branchdir')
        self.ls_equals('branchdir/subdir/\nbranchdir/subdir/b\n', 'branchdir --revision 1')

    def test_ls_ignored(self):
        self.wt.add(['a', '.bzrignore'])
        self.build_tree(['blah.py', 'blah.pyo', 'user-ignore'])
        self.ls_equals('.bzrignore\na\nblah.py\nblah.pyo\nuser-ignore\n')
        self.ls_equals('V        .bzrignore\nV        a\n?        blah.py\nI        blah.pyo\nI        user-ignore\n', '--verbose')
        self.ls_equals('blah.pyo\nuser-ignore\n', '--ignored')
        self.ls_equals('blah.py\n', '--unknown')
        self.ls_equals('.bzrignore\na\n', '--versioned')
        self.ls_equals('.bzrignore\na\n', '-V')

    def test_kinds(self):
        self.build_tree(['subdir/'])
        self.ls_equals('.bzrignore\na\n', '--kind=file')
        self.ls_equals('subdir/\n', '--kind=directory')
        self.ls_equals('', '--kind=symlink')
        self.run_bzr_error(['invalid kind specified'], 'ls --kind=pile')

    def test_ls_path_nonrecursive(self):
        self.ls_equals('%s/.bzrignore\n%s/a\n' % (self.test_dir, self.test_dir), self.test_dir, recursive=False)

    def test_ls_directory(self):
        """Test --directory option"""
        self.wt = self.make_branch_and_tree('dir')
        self.build_tree(['dir/sub/', 'dir/sub/file'])
        self.wt.add(['sub', 'sub/file'])
        self.wt.commit('commit')
        self.ls_equals('sub/\nsub/file\n', '--directory=dir')
        self.ls_equals('sub/file\n', '-d dir sub')