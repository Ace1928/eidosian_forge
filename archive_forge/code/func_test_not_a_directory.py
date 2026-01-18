import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_not_a_directory(self):
    filename = os.path.join(self.mkdtemp(), 'foo')
    self.touch(filename)
    mismatch = DirExists().match(filename)
    self.assertThat('%s is not a directory.' % filename, Equals(mismatch.describe()))