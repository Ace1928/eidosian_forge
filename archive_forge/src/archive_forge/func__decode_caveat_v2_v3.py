import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def _decode_caveat_v2_v3(version, key, caveat):
    """Decodes a version 2 or version 3 caveat.
    """
    if len(caveat) < 1 + _PUBLIC_KEY_PREFIX_LEN + _KEY_LEN + nacl.public.Box.NONCE_SIZE + 16:
        raise VerificationError('caveat id too short')
    original_caveat = caveat
    caveat = caveat[1:]
    pk_prefix = caveat[:_PUBLIC_KEY_PREFIX_LEN]
    caveat = caveat[_PUBLIC_KEY_PREFIX_LEN:]
    if key.public_key.serialize(raw=True)[:_PUBLIC_KEY_PREFIX_LEN] != pk_prefix:
        raise VerificationError('public key mismatch')
    first_party_pub = caveat[:_KEY_LEN]
    caveat = caveat[_KEY_LEN:]
    nonce = caveat[:nacl.public.Box.NONCE_SIZE]
    caveat = caveat[nacl.public.Box.NONCE_SIZE:]
    fp_public_key = nacl.public.PublicKey(first_party_pub)
    box = nacl.public.Box(key.key, fp_public_key)
    data = box.decrypt(caveat, nonce)
    root_key, condition, ns = _decode_secret_part_v2_v3(version, data)
    return ThirdPartyCaveatInfo(condition=condition.decode('utf-8'), first_party_public_key=PublicKey(fp_public_key), third_party_key_pair=key, root_key=root_key, caveat=original_caveat, version=version, id=None, namespace=ns)