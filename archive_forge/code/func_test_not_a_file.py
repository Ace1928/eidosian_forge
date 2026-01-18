import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_not_a_file(self):
    tempdir = self.mkdtemp()
    mismatch = FileExists().match(tempdir)
    self.assertThat('%s is not a file.' % tempdir, Equals(mismatch.describe()))