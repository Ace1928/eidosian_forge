from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
def test_to_utf8(self):
    self.assertEqual(encodeutils.to_utf8(b'a\xe9\xff'), b'a\xe9\xff')
    self.assertEqual(encodeutils.to_utf8('aéÿ€'), b'a\xc3\xa9\xc3\xbf\xe2\x82\xac')
    self.assertRaises(TypeError, encodeutils.to_utf8, 123)
    msg = oslo_i18n_fixture.Translation().lazy('test')
    self.assertEqual(encodeutils.to_utf8(msg), b'test')