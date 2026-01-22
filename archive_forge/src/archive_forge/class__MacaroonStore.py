import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
class _MacaroonStore(object):
    """ Stores root keys in memory and puts all operations in the macaroon id.
    """

    def __init__(self, key, locator):
        self._root_key_store = bakery.MemoryKeyStore()
        self._key = key
        self._locator = locator

    def new_macaroon(self, caveats, namespace, ops):
        root_key, id = self._root_key_store.root_key()
        m_id = {'id': base64.urlsafe_b64encode(id).decode('utf-8'), 'ops': ops}
        data = json.dumps(m_id)
        m = bakery.Macaroon(root_key=root_key, id=data, location='', version=bakery.LATEST_VERSION, namespace=namespace)
        m.add_caveats(caveats, self._key, self._locator)
        return m

    def macaroon_ops(self, ms):
        if len(ms) == 0:
            raise ValueError('no macaroons provided')
        m_id = json.loads(ms[0].identifier_bytes.decode('utf-8'))
        root_key = self._root_key_store.get(base64.urlsafe_b64decode(m_id['id'].encode('utf-8')))
        v = Verifier()

        class NoValidationOnFirstPartyCaveat(FirstPartyCaveatVerifierDelegate):

            def verify_first_party_caveat(self, verifier, caveat, signature):
                return True
        v.first_party_caveat_verifier_delegate = NoValidationOnFirstPartyCaveat()
        ok = v.verify(macaroon=ms[0], key=root_key, discharge_macaroons=ms[1:])
        if not ok:
            raise bakery.VerificationError('invalid signature')
        conditions = []
        for m in ms:
            cavs = m.first_party_caveats()
            for cav in cavs:
                conditions.append(cav.caveat_id_bytes.decode('utf-8'))
        ops = []
        for op in m_id['ops']:
            ops.append(bakery.Op(entity=op[0], action=op[1]))
        return (ops, conditions)