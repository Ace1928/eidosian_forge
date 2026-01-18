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
def test_ovf_process_no_config_file(self, mock_find_file):
    mock_find_file.return_value = None
    ova_file_path = self._copy_ova_to_tmpdir('testserver.ova')
    ova_uri = 'file://' + ova_file_path
    oprocess = ovf_process._OVF_Process('task_id', 'ovf_proc', self.img_repo)
    self.assertEqual(ova_uri, oprocess.execute('test_image_id', ova_uri))
    with open(ova_file_path, 'rb') as disk_image_file:
        content = disk_image_file.read()
    self.assertEqual(b'ABCD', content)
    self.image.extra_properties.update.assert_called_once_with({})
    self.assertEqual('bare', self.image.container_format)