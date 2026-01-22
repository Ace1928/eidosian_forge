import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def _decode_caveat_v1(key, caveat):
    """Decode a base64 encoded JSON id.

    @param key the nacl private key to decode.
    @param caveat a base64 encoded JSON string.
    """
    data = base64.b64decode(caveat).decode('utf-8')
    wrapper = json.loads(data)
    tp_public_key = nacl.public.PublicKey(base64.b64decode(wrapper['ThirdPartyPublicKey']))
    if key.public_key.key != tp_public_key:
        raise Exception('public key mismatch')
    if wrapper.get('FirstPartyPublicKey', None) is None:
        raise Exception('target service public key not specified')
    secret = base64.b64decode(wrapper.get('Id'))
    nonce = base64.b64decode(wrapper.get('Nonce'))
    fp_public_key = nacl.public.PublicKey(base64.b64decode(wrapper.get('FirstPartyPublicKey')))
    box = nacl.public.Box(key.key, fp_public_key)
    c = box.decrypt(secret, nonce)
    record = json.loads(c.decode('utf-8'))
    fp_key = nacl.public.PublicKey(base64.b64decode(wrapper.get('FirstPartyPublicKey')))
    return ThirdPartyCaveatInfo(condition=record.get('Condition'), first_party_public_key=PublicKey(fp_key), third_party_key_pair=key, root_key=base64.b64decode(record.get('RootKey')), caveat=caveat, id=None, version=VERSION_1, namespace=legacy_namespace())