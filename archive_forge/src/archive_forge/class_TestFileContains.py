import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class TestFileContains(TestCase, PathHelpers):

    def test_not_exists(self):
        doesntexist = os.path.join(self.mkdtemp(), 'doesntexist')
        mismatch = FileContains('').match(doesntexist)
        self.assertThat(PathExists().match(doesntexist).describe(), Equals(mismatch.describe()))

    def test_contains(self):
        tempdir = self.mkdtemp()
        filename = os.path.join(tempdir, 'foo')
        self.create_file(filename, 'Hello World!')
        self.assertThat(filename, FileContains('Hello World!'))

    def test_matcher(self):
        tempdir = self.mkdtemp()
        filename = os.path.join(tempdir, 'foo')
        self.create_file(filename, 'Hello World!')
        self.assertThat(filename, FileContains(matcher=DocTestMatches('Hello World!')))

    def test_neither_specified(self):
        self.assertRaises(AssertionError, FileContains)

    def test_both_specified(self):
        self.assertRaises(AssertionError, FileContains, contents=[], matcher=Contains('a'))

    def test_does_not_contain(self):
        tempdir = self.mkdtemp()
        filename = os.path.join(tempdir, 'foo')
        self.create_file(filename, 'Goodbye Cruel World!')
        mismatch = FileContains('Hello World!').match(filename)
        self.assertThat(Equals('Hello World!').match('Goodbye Cruel World!').describe(), Equals(mismatch.describe()))