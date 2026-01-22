import unittest
from subprocess import Popen, PIPE, STDOUT
import time
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
class TestWithOVS12(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mn = Mininet()
        c = cls.mn.addController(controller=RemoteController, ip=OSKEN_HOST, port=OSKEN_PORT)
        c.start()
        s1 = cls.mn.addSwitch('s1', cls=OVS12KernelSwitch)
        s1.start(cls.mn.controllers)
        h1 = cls.mn.addHost('h1', ip='0.0.0.0/0')
        link = cls.mn.addLink(h1, s1)
        s1.attach(link.intf2)

    @classmethod
    def tearDownClass(cls):
        cls.mn.stop()

    def test_add_flow_v10(self):
        app = 'os_ken/tests/integrated/test_add_flow_v10.py'
        self._run_os_ken_manager_and_check_output(app)

    def test_request_reply_v12(self):
        app = 'os_ken/tests/integrated/test_request_reply_v12.py'
        self._run_os_ken_manager_and_check_output(app)

    def test_add_flow_v12_actions(self):
        app = 'os_ken/tests/integrated/test_add_flow_v12_actions.py'
        self._run_os_ken_manager_and_check_output(app)

    def test_add_flow_v12_matches(self):
        app = 'os_ken/tests/integrated/test_add_flow_v12_matches.py'
        self._run_os_ken_manager_and_check_output(app)

    def test_of_config(self):
        self.skipTest('OVS 1.10 does not support of_config')

    def _run_os_ken_manager_and_check_output(self, app):
        cmd = [PYTHON_BIN, OSKEN_MGR, app]
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT)
        while True:
            if p.poll() is not None:
                raise Exception('Another osken-manager already running?')
            line = p.stdout.readline().strip()
            if line == '':
                time.sleep(1)
                continue
            print('osken-manager: %s' % line)
            if line.find('TEST_FINISHED') != -1:
                self.assertTrue(line.find('Completed=[True]') != -1)
                p.terminate()
                p.communicate()
                break