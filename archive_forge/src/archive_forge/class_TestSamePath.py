import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class TestSamePath(TestCase, PathHelpers):

    def test_same_string(self):
        self.assertThat('foo', SamePath('foo'))

    def test_relative_and_absolute(self):
        path = 'foo'
        abspath = os.path.abspath(path)
        self.assertThat(path, SamePath(abspath))
        self.assertThat(abspath, SamePath(path))

    def test_real_path(self):
        tempdir = self.mkdtemp()
        source = os.path.join(tempdir, 'source')
        self.touch(source)
        target = os.path.join(tempdir, 'target')
        try:
            os.symlink(source, target)
        except (AttributeError, NotImplementedError):
            self.skipTest('No symlink support')
        self.assertThat(source, SamePath(target))
        self.assertThat(target, SamePath(source))