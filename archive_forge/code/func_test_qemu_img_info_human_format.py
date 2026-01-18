import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
@mock.patch('debtcollector.deprecate')
def test_qemu_img_info_human_format(self, mock_deprecate):
    img_info = self._initialize_img_info()
    img_info = img_info + ('cluster_size: %s' % self.cluster_size,)
    if self.backing_file is not None:
        img_info = img_info + ('backing file: %s' % self.backing_file,)
        if self.backing_file_format is not None:
            img_info = img_info + ('backing file format: %s' % self.backing_file_format,)
    if self.encrypted is not None:
        img_info = img_info + ('encrypted: %s' % self.encrypted,)
    if self.garbage_before_snapshot is True:
        img_info = img_info + ('blah BLAH: bb',)
    if self.snapshot_count is not None:
        img_info = self._insert_snapshots(img_info)
    if self.garbage_before_snapshot is False:
        img_info = img_info + ('junk stuff: bbb',)
    example_output = '\n'.join(img_info)
    warnings.simplefilter('always', FutureWarning)
    image_info = imageutils.QemuImgInfo(example_output)
    mock_deprecate.assert_called()
    self._base_validation(image_info)
    self.assertEqual(image_info.cluster_size, self.exp_cluster_size)
    if self.backing_file is not None:
        self.assertEqual(image_info.backing_file, self.exp_backing_file)
        if self.backing_file_format is not None:
            self.assertEqual(image_info.backing_file_format, self.exp_backing_file_format)
    if self.encrypted is not None:
        self.assertEqual(image_info.encrypted, self.encrypted)