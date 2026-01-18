import os.path
import shutil
import tarfile
import tempfile
from unittest import mock
from defusedxml.ElementTree import ParseError
from glance.async_.flows import ovf_process
import glance.tests.utils as test_utils
from oslo_config import cfg
@mock.patch.object(cfg.ConfigOpts, 'find_file')
def test_ovf_process_not_admin(self, mock_find_file):
    mock_find_file.return_value = self.config_file_name
    ova_file_path = self._copy_ova_to_tmpdir('testserver.ova')
    ova_uri = 'file://' + ova_file_path
    self.image.context.is_admin = False
    oprocess = ovf_process._OVF_Process('task_id', 'ovf_proc', self.img_repo)
    self.assertRaises(RuntimeError, oprocess.execute, 'test_image_id', ova_uri)