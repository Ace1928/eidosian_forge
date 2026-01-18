import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
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