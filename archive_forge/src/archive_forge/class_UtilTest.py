from six.moves import range
import sys
from pyu2f.tests.lib import util
class UtilTest(unittest.TestCase):

    def testSimplePing(self):
        dev = util.FakeHidDevice(cid_to_allocate=None)
        dev.Write([0, 0, 0, 1, 129, 0, 3, 1, 2, 3])
        self.assertEquals(dev.Read(), [0, 0, 0, 1, 129, 0, 3, 1, 2, 3] + [0 for _ in range(54)])

    def testErrorBusy(self):
        dev = util.FakeHidDevice(cid_to_allocate=None)
        dev.SetChannelBusyCount(2)
        dev.Write([0, 0, 0, 1, 129, 0, 3, 1, 2, 3])
        self.assertEquals(dev.Read(), [0, 0, 0, 1, 191, 0, 1, 6] + [0 for _ in range(56)])
        dev.Write([0, 0, 0, 1, 129, 0, 3, 1, 2, 3])
        self.assertEquals(dev.Read(), [0, 0, 0, 1, 191, 0, 1, 6] + [0 for _ in range(56)])
        dev.Write([0, 0, 0, 1, 129, 0, 3, 1, 2, 3])
        self.assertEquals(dev.Read(), [0, 0, 0, 1, 129, 0, 3, 1, 2, 3] + [0 for _ in range(54)])

    def testFragmentedApdu(self):
        dev = util.FakeHidDevice(cid_to_allocate=None, msg_reply=list(range(85, 0, -1)))
        dev.Write([0, 0, 0, 1, 131, 0, 100] + [x for x in range(57)])
        dev.Write([0, 0, 0, 1, 0] + [x for x in range(57, 100)])
        self.assertEquals(dev.Read(), [0, 0, 0, 1, 131, 0, 85] + [x for x in range(85, 28, -1)])
        self.assertEquals(dev.Read(), [0, 0, 0, 1, 0] + [x for x in range(28, 0, -1)] + [0 for _ in range(31)])