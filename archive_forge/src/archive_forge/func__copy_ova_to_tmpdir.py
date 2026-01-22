import os.path
import shutil
import tarfile
import tempfile
from unittest import mock
from defusedxml.ElementTree import ParseError
from glance.async_.flows import ovf_process
import glance.tests.utils as test_utils
from oslo_config import cfg
def _copy_ova_to_tmpdir(self, ova_name):
    shutil.copy(os.path.join(self.test_ova_dir, ova_name), self.tempdir)
    return os.path.join(self.tempdir, ova_name)