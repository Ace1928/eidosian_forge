from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_ok_TODO_xfail_comment(self):
    self.tap.write(_u('ok # TODO Not done yet\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.check_events([('status', 'test 1', 'xfail', None, False, 'tap comment', b'Not done yet', True, 'text/plain; charset=UTF8', None, None)])