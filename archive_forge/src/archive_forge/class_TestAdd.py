import contextlib
import errno
import os
import resource
import sys
from breezy import osutils, tests
from breezy.tests import features, script
class TestAdd(tests.TestCaseWithTransport):

    def writeBigFile(self, path):
        self.addCleanup(os.unlink, path)
        try:
            make_big_file(path)
        except OSError as e:
            if e.errno == errno.ENOSPC:
                self.skipTest('not enough disk space for big file')

    def setUp(self):
        super().setUp()
        cm = limit_memory(LIMIT)
        try:
            cm.__enter__()
        except NotImplementedError:
            self.skipTest('memory limits not supported on this platform')
        self.addCleanup(cm.__exit__, None, None, None)

    def test_allocate(self):

        def allocate():
            '.' * BIG_FILE_SIZE
        self.assertRaises(MemoryError, allocate)

    def test_add(self):
        tree = self.make_branch_and_tree('tree')
        self.writeBigFile(os.path.join(tree.basedir, 'testfile'))
        tree.add('testfile')

    def test_make_file_big(self):
        self.knownFailure('commit keeps entire files in memory')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/testfile'])
        tree.add('testfile')
        tree.commit('add small file')
        self.writeBigFile(os.path.join(tree.basedir, 'testfile'))
        tree.commit('small files get big')
        self.knownFailure('commit keeps entire files in memory')

    def test_commit(self):
        self.knownFailure('commit keeps entire files in memory')
        tree = self.make_branch_and_tree('tree')
        self.writeBigFile(os.path.join(tree.basedir, 'testfile'))
        tree.add('testfile')
        tree.commit('foo')

    def test_clone(self):
        self.knownFailure('commit keeps entire files in memory')
        tree = self.make_branch_and_tree('tree')
        self.writeBigFile(os.path.join(tree.basedir, 'testfile'))
        tree.add('testfile')
        tree.commit('foo')
        tree.clone.sprout('newtree')