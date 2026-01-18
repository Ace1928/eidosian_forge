import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_third_party_discharge_macaroon_wrong_root_key_and_third_party_caveat(self):
    root_keys = bakery.MemoryKeyStore()
    ts = bakery.Bakery(key=bakery.generate_key(), checker=common.test_checker(), root_key_store=root_keys, identity_client=common.OneIdentity())
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    ts_macaroon = ts.oven.macaroon(bakery.LATEST_VERSION, common.ages, None, [bakery.LOGIN_OP])
    ts_macaroon.add_caveat(checkers.Caveat(location='bs-loc', condition='true'), ts.oven.key, locator)

    def get_discharge(cav, payload):
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, bs.oven.key, common.ThirdPartyStrcmpChecker('true'), bs.oven.locator)
    d = bakery.discharge_all(ts_macaroon, get_discharge)
    ts.checker.auth([d]).allow(common.test_context, [bakery.LOGIN_OP])
    root_keys._key = os.urandom(24)
    with self.assertRaises(bakery.PermissionDenied) as err:
        ts.checker.auth([d]).allow(common.test_context, [bakery.LOGIN_OP])
    self.assertEqual(str(err.exception), 'verification failed: Decryption failed. Ciphertext failed verification')