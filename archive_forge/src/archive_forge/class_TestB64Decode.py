import json
from datetime import datetime
from unittest import TestCase
import macaroonbakery.bakery as bakery
import pymacaroons
from macaroonbakery._utils import cookie
from pymacaroons.serializers import json_serializer
class TestB64Decode(TestCase):

    def test_decode(self):
        test_cases = [{'about': 'empty string', 'input': '', 'expect': ''}, {'about': 'standard encoding, padded', 'input': 'Z29+IQ==', 'expect': 'go~!'}, {'about': 'URL encoding, padded', 'input': 'Z29-IQ==', 'expect': 'go~!'}, {'about': 'standard encoding, not padded', 'input': 'Z29+IQ', 'expect': 'go~!'}, {'about': 'URL encoding, not padded', 'input': 'Z29-IQ', 'expect': 'go~!'}, {'about': 'standard encoding, not enough much padding', 'input': 'Z29+IQ=', 'expect_error': 'illegal base64 data at input byte 8'}]
        for test in test_cases:
            if test.get('expect_error'):
                with self.assertRaises(ValueError, msg=test['about']) as e:
                    bakery.b64decode(test['input'])
                self.assertEqual(str(e.exception), 'Incorrect padding')
            else:
                self.assertEqual(bakery.b64decode(test['input']), test['expect'].encode('utf-8'), msg=test['about'])