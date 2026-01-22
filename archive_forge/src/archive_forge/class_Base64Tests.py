from oslo_serialization import base64
from oslotest import base as test_base
class Base64Tests(test_base.BaseTestCase):

    def test_encode_as_bytes(self):
        self.assertEqual(b'dGV4dA==', base64.encode_as_bytes(b'text'))
        self.assertEqual(b'dGV4dA==', base64.encode_as_bytes('text'))
        self.assertEqual(b'ZTrDqQ==', base64.encode_as_bytes('e:é'))
        self.assertEqual(b'ZTrp', base64.encode_as_bytes('e:é', encoding='latin1'))

    def test_encode_as_text(self):
        self.assertEqual('dGV4dA==', base64.encode_as_text(b'text'))
        self.assertEqual('dGV4dA==', base64.encode_as_text('text'))
        self.assertEqual('ZTrDqQ==', base64.encode_as_text('e:é'))
        self.assertEqual('ZTrp', base64.encode_as_text('e:é', encoding='latin1'))

    def test_decode_as_bytes(self):
        self.assertEqual(b'text', base64.decode_as_bytes(b'dGV4dA=='))
        self.assertEqual(b'text', base64.decode_as_bytes('dGV4dA=='))

    def test_decode_as_bytes__error(self):
        self.assertRaises(TypeError, base64.decode_as_bytes, 'hello world')

    def test_decode_as_text(self):
        self.assertEqual('text', base64.decode_as_text(b'dGV4dA=='))
        self.assertEqual('text', base64.decode_as_text('dGV4dA=='))
        self.assertEqual('e:é', base64.decode_as_text('ZTrDqQ=='))
        self.assertEqual('e:é', base64.decode_as_text('ZTrp', encoding='latin1'))