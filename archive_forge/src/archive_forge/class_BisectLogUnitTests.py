import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
class BisectLogUnitTests(BisectTestCase):
    """Test the BisectLog class."""

    def testCreateBlank(self):
        """Test creation of new log."""
        bisect_log = bisect.BisectLog(self.tree.controldir)
        bisect_log.save()
        self.assertTrue(os.path.exists(os.path.join('.bzr', bisect.BISECT_INFO_PATH)))

    def testLoad(self):
        """Test loading a log."""
        preloaded_log = open(os.path.join('.bzr', bisect.BISECT_INFO_PATH), 'w')
        preloaded_log.write('rev1 yes\nrev2 no\nrev3 yes\n')
        preloaded_log.close()
        bisect_log = bisect.BisectLog(self.tree.controldir)
        self.assertEqual(len(bisect_log._items), 3)
        self.assertEqual(bisect_log._items[0], (b'rev1', 'yes'))
        self.assertEqual(bisect_log._items[1], (b'rev2', 'no'))
        self.assertEqual(bisect_log._items[2], (b'rev3', 'yes'))

    def testSave(self):
        """Test saving the log."""
        bisect_log = bisect.BisectLog(self.tree.controldir)
        bisect_log._items = [(b'rev1', 'yes'), (b'rev2', 'no'), (b'rev3', 'yes')]
        bisect_log.save()
        with open(os.path.join('.bzr', bisect.BISECT_INFO_PATH), 'rb') as logfile:
            self.assertEqual(logfile.read(), b'rev1 yes\nrev2 no\nrev3 yes\n')