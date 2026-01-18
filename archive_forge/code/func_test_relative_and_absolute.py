import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def test_relative_and_absolute(self):
    path = 'foo'
    abspath = os.path.abspath(path)
    self.assertThat(path, SamePath(abspath))
    self.assertThat(abspath, SamePath(path))