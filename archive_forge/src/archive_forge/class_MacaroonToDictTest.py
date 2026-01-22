import json
from datetime import datetime
from unittest import TestCase
import macaroonbakery.bakery as bakery
import pymacaroons
from macaroonbakery._utils import cookie
from pymacaroons.serializers import json_serializer
class MacaroonToDictTest(TestCase):

    def test_macaroon_to_dict(self):
        m = pymacaroons.Macaroon(key=b'rootkey', identifier=b'some id', location='here', version=2)
        as_dict = bakery.macaroon_to_dict(m)
        data = json.dumps(as_dict)
        m1 = pymacaroons.Macaroon.deserialize(data, json_serializer.JsonSerializer())
        self.assertEqual(m1.signature, m.signature)
        pymacaroons.Verifier().verify(m1, b'rootkey')