import os
from oslo_vmware import image_util
from oslo_vmware.tests import base
class ImageUtilTest(base.TestCase):

    def test_get_vmdk_name_from_ovf(self):
        ovf_descriptor = os.path.join(os.path.dirname(__file__), 'test.ovf')
        with open(ovf_descriptor) as f:
            vmdk_name = image_util.get_vmdk_name_from_ovf(f)
            self.assertEqual('test-disk1.vmdk', vmdk_name)