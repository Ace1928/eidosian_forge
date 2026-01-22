from taskflow import test
from taskflow.utils import misc
class BinaryEncodeTest(test.TestCase):

    def _check(self, data, expected_result):
        result = misc.binary_encode(data)
        self.assertIsInstance(result, bytes)
        self.assertEqual(expected_result, result)

    def test_simple_binary(self):
        data = _bytes('hello')
        self._check(data, data)

    def test_unicode_binary(self):
        data = _bytes('привет')
        self._check(data, data)

    def test_simple_text(self):
        self._check(u'hello', _bytes('hello'))

    def test_unicode_text(self):
        self._check(u'привет', _bytes('привет'))

    def test_unicode_other_encoding(self):
        result = misc.binary_encode(u'mañana', 'latin-1')
        self.assertIsInstance(result, bytes)
        self.assertEqual(u'mañana'.encode('latin-1'), result)