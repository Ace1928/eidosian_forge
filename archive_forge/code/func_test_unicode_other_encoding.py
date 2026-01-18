from taskflow import test
from taskflow.utils import misc
def test_unicode_other_encoding(self):
    data = u'mañana'.encode('latin-1')
    result = misc.binary_decode(data, 'latin-1')
    self.assertIsInstance(result, str)
    self.assertEqual(u'mañana', result)