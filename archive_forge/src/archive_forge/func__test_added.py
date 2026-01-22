import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def _test_added(self, name, output, null=False):

    def check_added(expected, null=False):
        command = 'added'
        if null:
            command += ' --null'
        out, err = self.run_bzr(command)
        self.assertEqual(out, expected)
        self.assertEqual(err, '')
    tree = self.make_branch_and_tree('.')
    check_added('')
    self.build_tree_contents([(name, b'contents of %s\n' % (name.encode('utf-8'),))])
    check_added('')
    tree.add(name)
    check_added(output, null)
    tree.commit(message='add "%s"' % name)
    check_added('')