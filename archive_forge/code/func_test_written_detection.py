from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
def test_written_detection(self):
    w = Weave()
    w.add_lines(b'v1', [], [b'hello\n'])
    w.add_lines(b'v2', [b'v1'], [b'hello\n', b'there\n'])
    tmpf = BytesIO()
    write_weave(w, tmpf)
    self.assertEqual(b'# bzr weave file v5\ni\n1 f572d396fae9206628714fb2ce00f72e94f2258f\nn v1\n\ni 0\n1 90f265c6e75f1c8f9ab76dcf85528352c5f215ef\nn v2\n\nw\n{ 0\n. hello\n}\n{ 1\n. there\n}\nW\n', tmpf.getvalue())
    tmpf = BytesIO(b'# bzr weave file v5\ni\n1 f572d396fae9206628714fb2ce00f72e94f2258f\nn v1\n\ni 0\n1 90f265c6e75f1c8f9ab76dcf85528352c5f215ef\nn v2\n\nw\n{ 0\n. hello\n}\n{ 1\n. There\n}\nW\n')
    w = read_weave(tmpf)
    self.assertEqual(b'hello\n', w.get_text(b'v1'))
    self.assertRaises(WeaveInvalidChecksum, w.get_text, b'v2')
    self.assertRaises(WeaveInvalidChecksum, w.get_lines, b'v2')
    self.assertRaises(WeaveInvalidChecksum, w.check)
    tmpf = BytesIO(b'# bzr weave file v5\ni\n1 f572d396fae9206628714fb2ce00f72e94f2258f\nn v1\n\ni 0\n1 f0f265c6e75f1c8f9ab76dcf85528352c5f215ef\nn v2\n\nw\n{ 0\n. hello\n}\n{ 1\n. there\n}\nW\n')
    w = read_weave(tmpf)
    self.assertEqual(b'hello\n', w.get_text(b'v1'))
    self.assertRaises(WeaveInvalidChecksum, w.get_text, b'v2')
    self.assertRaises(WeaveInvalidChecksum, w.get_lines, b'v2')
    self.assertRaises(WeaveInvalidChecksum, w.check)