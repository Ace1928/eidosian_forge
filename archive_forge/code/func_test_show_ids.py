from breezy import ignores, tests
def test_show_ids(self):
    self.build_tree(['subdir/'])
    self.wt.add(['a', 'subdir'], ids=[b'a-id', b'subdir-id'])
    self.ls_equals('.bzrignore                                         \na                                                  a-id\nsubdir/                                            subdir-id\n', '--show-ids')
    self.ls_equals('?        .bzrignore\nV        a                                         a-id\nV        subdir/                                   subdir-id\n', '--show-ids --verbose')
    self.ls_equals('.bzrignore\x00\x00a\x00a-id\x00subdir\x00subdir-id\x00', '--show-ids --null')