import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def _test_json_with_version(self, version):
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    ns = checkers.Namespace({'testns': 'x'})
    m = bakery.Macaroon(root_key=b'root key', id=b'id', location='location', version=version, namespace=ns)
    m.add_caveat(checkers.Caveat(location='bs-loc', condition='something'), bs.oven.key, locator)
    self.assertEqual(len(m._caveat_data), 0)
    data = json.dumps(m, cls=bakery.MacaroonJSONEncoder)
    m1 = json.loads(data, cls=bakery.MacaroonJSONDecoder)
    self.assertEqual(m1.macaroon.signature, m.macaroon.signature)
    self.assertEqual(m1.macaroon.version, bakery.macaroon_version(version))
    self.assertEqual(len(m1.macaroon.caveats), 1)
    self.assertEqual(m1.namespace, bakery.legacy_namespace())
    self.assertEqual(len(m1._caveat_data), 0)