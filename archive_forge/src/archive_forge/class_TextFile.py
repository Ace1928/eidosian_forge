from io import BytesIO
from ..errors import BinaryFile
from ..textfile import check_text_lines, check_text_path, text_file
from . import TestCase, TestCaseInTempDir
class TextFile(TestCase):

    def test_text_file(self):
        s = BytesIO(b'ab' * 2048)
        self.assertEqual(text_file(s).read(), s.getvalue())
        s = BytesIO(b'a' * 1023 + b'\x00')
        self.assertRaises(BinaryFile, text_file, s)
        s = BytesIO(b'a' * 1024 + b'\x00')
        self.assertEqual(text_file(s).read(), s.getvalue())

    def test_check_text_lines(self):
        lines = [b'ab' * 2048]
        check_text_lines(lines)
        lines = [b'a' * 1023 + b'\x00']
        self.assertRaises(BinaryFile, check_text_lines, lines)