from unittest import mock
from os_win import utilsfactory
from os_brick.tests import base
class WindowsConnectorTestBase(base.TestCase):

    @mock.patch('sys.platform', 'win32')
    def setUp(self):
        super(WindowsConnectorTestBase, self).setUp()
        utilsfactory_patcher = mock.patch.object(utilsfactory, '_get_class')
        utilsfactory_patcher.start()
        self.addCleanup(utilsfactory_patcher.stop)