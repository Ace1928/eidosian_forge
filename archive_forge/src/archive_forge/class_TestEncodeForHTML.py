import unittest
import simplejson as json
class TestEncodeForHTML(unittest.TestCase):

    def setUp(self):
        self.decoder = json.JSONDecoder()
        self.encoder = json.JSONEncoderForHTML()
        self.non_ascii_encoder = json.JSONEncoderForHTML(ensure_ascii=False)

    def test_basic_encode(self):
        self.assertEqual('"\\u0026"', self.encoder.encode('&'))
        self.assertEqual('"\\u003c"', self.encoder.encode('<'))
        self.assertEqual('"\\u003e"', self.encoder.encode('>'))
        self.assertEqual('"\\u2028"', self.encoder.encode(u'\u2028'))

    def test_non_ascii_basic_encode(self):
        self.assertEqual('"\\u0026"', self.non_ascii_encoder.encode('&'))
        self.assertEqual('"\\u003c"', self.non_ascii_encoder.encode('<'))
        self.assertEqual('"\\u003e"', self.non_ascii_encoder.encode('>'))
        self.assertEqual('"\\u2028"', self.non_ascii_encoder.encode(u'\u2028'))

    def test_basic_roundtrip(self):
        for char in '&<>':
            self.assertEqual(char, self.decoder.decode(self.encoder.encode(char)))

    def test_prevent_script_breakout(self):
        bad_string = '</script><script>alert("gotcha")</script>'
        self.assertEqual('"\\u003c/script\\u003e\\u003cscript\\u003ealert(\\"gotcha\\")\\u003c/script\\u003e"', self.encoder.encode(bad_string))
        self.assertEqual(bad_string, self.decoder.decode(self.encoder.encode(bad_string)))