from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
def test_pxe_1_19(self):
    """Check baremetal port create command with PXE option.

        Test steps:
        1) Create port using --pxe-enabled argument.
        2) Check that port successfully created with right PXE option.
        """
    pxe_values = [True, False]
    api_version = ' --os-baremetal-api-version 1.19'
    for value in pxe_values:
        port = self.port_create(self.node['uuid'], params='--pxe-enabled {0} {1}'.format(value, api_version))
        self.assertEqual(value, port['pxe_enabled'])