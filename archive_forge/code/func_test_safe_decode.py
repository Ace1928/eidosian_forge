from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
def test_safe_decode(self):
    safe_decode = encodeutils.safe_decode
    self.assertRaises(TypeError, safe_decode, True)
    self.assertEqual('niño', safe_decode('niÃ±o'.encode('latin-1'), incoming='utf-8'))
    self.assertEqual('strange', safe_decode('\x80strange'.encode('latin-1'), errors='ignore'))
    self.assertEqual('À', safe_decode('À'.encode('latin-1'), incoming='iso-8859-1'))
    self.assertEqual('niño', safe_decode('niÃ±o'.encode('latin-1'), incoming='ascii'))
    self.assertEqual('foo', safe_decode(b'foo'))