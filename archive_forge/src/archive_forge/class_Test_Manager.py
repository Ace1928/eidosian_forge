import unittest
from unittest import mock
from importlib import reload
from os_ken.cmd.manager import main
class Test_Manager(unittest.TestCase):
    """Test osken-manager command
    """

    def __init__(self, methodName):
        super(Test_Manager, self).__init__(methodName)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch('sys.argv', new=['osken-manager', '--version'])
    def test_version(self):
        self.assertRaises(SystemExit, main)

    @mock.patch('sys.argv', new=['osken-manager', '--help'])
    def test_help(self):
        self.assertRaises(SystemExit, main)

    @staticmethod
    def _reset_globals():
        import os_ken.base.app_manager
        import os_ken.ofproto.ofproto_protocol
        reload(os_ken.base.app_manager)
        reload(os_ken.ofproto.ofproto_protocol)

    @mock.patch('sys.argv', new=['osken-manager', '--verbose', 'os_ken.tests.unit.cmd.dummy_app'])
    def test_no_services(self):
        self._reset_globals()
        main()
        self._reset_globals()

    @mock.patch('sys.argv', new=['osken-manager', '--verbose', 'os_ken.tests.unit.cmd.dummy_openflow_app'])
    def test_openflow_app(self):
        self._reset_globals()
        main()
        self._reset_globals()