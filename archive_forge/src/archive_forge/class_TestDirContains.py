import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class TestDirContains(TestCase, PathHelpers):

    def test_empty(self):
        tempdir = self.mkdtemp()
        self.assertThat(tempdir, DirContains([]))

    def test_not_exists(self):
        doesntexist = os.path.join(self.mkdtemp(), 'doesntexist')
        mismatch = DirContains([]).match(doesntexist)
        self.assertThat(PathExists().match(doesntexist).describe(), Equals(mismatch.describe()))

    def test_contains_files(self):
        tempdir = self.mkdtemp()
        self.touch(os.path.join(tempdir, 'foo'))
        self.touch(os.path.join(tempdir, 'bar'))
        self.assertThat(tempdir, DirContains(['bar', 'foo']))

    def test_matcher(self):
        tempdir = self.mkdtemp()
        self.touch(os.path.join(tempdir, 'foo'))
        self.touch(os.path.join(tempdir, 'bar'))
        self.assertThat(tempdir, DirContains(matcher=Contains('bar')))

    def test_neither_specified(self):
        self.assertRaises(AssertionError, DirContains)

    def test_both_specified(self):
        self.assertRaises(AssertionError, DirContains, filenames=[], matcher=Contains('a'))

    def test_does_not_contain_files(self):
        tempdir = self.mkdtemp()
        self.touch(os.path.join(tempdir, 'foo'))
        mismatch = DirContains(['bar', 'foo']).match(tempdir)
        self.assertThat(Equals(['bar', 'foo']).match(['foo']).describe(), Equals(mismatch.describe()))