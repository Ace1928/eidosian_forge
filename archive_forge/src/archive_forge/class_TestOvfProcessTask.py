import os.path
import shutil
import tarfile
import tempfile
from unittest import mock
from defusedxml.ElementTree import ParseError
from glance.async_.flows import ovf_process
import glance.tests.utils as test_utils
from oslo_config import cfg
class TestOvfProcessTask(test_utils.BaseTestCase):

    def setUp(self):
        super(TestOvfProcessTask, self).setUp()
        self.test_ova_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'var'))
        self.tempdir = tempfile.mkdtemp()
        self.config(work_dir=self.tempdir, group='task')
        interested_properties = '{\n   "cim_pasd":  [\n      "InstructionSetExtensionName",\n      "ProcessorArchitecture"]\n}\n'
        self.config_file_name = os.path.join(self.tempdir, 'ovf-metadata.json')
        with open(self.config_file_name, 'w') as config_file:
            config_file.write(interested_properties)
        self.image = mock.Mock()
        self.image.container_format = 'ova'
        self.image.context.is_admin = True
        self.img_repo = mock.Mock()
        self.img_repo.get.return_value = self.image

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
        super(TestOvfProcessTask, self).tearDown()

    def _copy_ova_to_tmpdir(self, ova_name):
        shutil.copy(os.path.join(self.test_ova_dir, ova_name), self.tempdir)
        return os.path.join(self.tempdir, ova_name)

    @mock.patch.object(cfg.ConfigOpts, 'find_file')
    def test_ovf_process_success(self, mock_find_file):
        mock_find_file.return_value = self.config_file_name
        ova_file_path = self._copy_ova_to_tmpdir('testserver.ova')
        ova_uri = 'file://' + ova_file_path
        oprocess = ovf_process._OVF_Process('task_id', 'ovf_proc', self.img_repo)
        self.assertEqual(ova_uri, oprocess.execute('test_image_id', ova_uri))
        with open(ova_file_path, 'rb') as disk_image_file:
            content = disk_image_file.read()
        self.assertEqual(b'ABCD', content)
        self.image.extra_properties.update.assert_called_once_with({'cim_pasd_InstructionSetExtensionName': 'DMTF:x86:VT-d'})
        self.assertEqual('bare', self.image.container_format)

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

    @mock.patch.object(cfg.ConfigOpts, 'find_file')
    def test_ovf_process_not_admin(self, mock_find_file):
        mock_find_file.return_value = self.config_file_name
        ova_file_path = self._copy_ova_to_tmpdir('testserver.ova')
        ova_uri = 'file://' + ova_file_path
        self.image.context.is_admin = False
        oprocess = ovf_process._OVF_Process('task_id', 'ovf_proc', self.img_repo)
        self.assertRaises(RuntimeError, oprocess.execute, 'test_image_id', ova_uri)

    def test_extract_ova_not_tar(self):
        ova_file_path = os.path.join(self.test_ova_dir, 'testserver-not-tar.ova')
        iextractor = ovf_process.OVAImageExtractor()
        with open(ova_file_path, 'rb') as ova_file:
            self.assertRaises(tarfile.ReadError, iextractor.extract, ova_file)

    def test_extract_ova_no_disk(self):
        ova_file_path = os.path.join(self.test_ova_dir, 'testserver-no-disk.ova')
        iextractor = ovf_process.OVAImageExtractor()
        with open(ova_file_path, 'rb') as ova_file:
            self.assertRaises(KeyError, iextractor.extract, ova_file)

    def test_extract_ova_no_ovf(self):
        ova_file_path = os.path.join(self.test_ova_dir, 'testserver-no-ovf.ova')
        iextractor = ovf_process.OVAImageExtractor()
        with open(ova_file_path, 'rb') as ova_file:
            self.assertRaises(RuntimeError, iextractor.extract, ova_file)

    def test_extract_ova_bad_ovf(self):
        ova_file_path = os.path.join(self.test_ova_dir, 'testserver-bad-ovf.ova')
        iextractor = ovf_process.OVAImageExtractor()
        with open(ova_file_path, 'rb') as ova_file:
            self.assertRaises(ParseError, iextractor._parse_OVF, ova_file)