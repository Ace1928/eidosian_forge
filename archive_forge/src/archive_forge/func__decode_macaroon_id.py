import base64
import hashlib
import itertools
import os
import google
from ._checker import (Op, LOGIN_OP)
from ._store import MemoryKeyStore
from ._error import VerificationError
from ._versions import (
from ._macaroon import (
import macaroonbakery.checkers as checkers
import six
from macaroonbakery._utils import (
from ._internal import id_pb2
from pymacaroons import MACAROON_V2, Verifier
def _decode_macaroon_id(id):
    storage_id = b''
    base64_decoded = False
    first = id[:1]
    if first == b'A':
        try:
            dec = b64decode(id.decode('utf-8'))
            id = dec
            base64_decoded = True
        except:
            pass
    first = six.byte2int(id[:1])
    if first == VERSION_2:
        storage_id = id[1 + 16:]
    if first == VERSION_3:
        try:
            id1 = id_pb2.MacaroonId.FromString(id[1:])
        except google.protobuf.message.DecodeError:
            raise VerificationError('no operations found in macaroon')
        if len(id1.ops) == 0 or len(id1.ops[0].actions) == 0:
            raise VerificationError('no operations found in macaroon')
        ops = []
        for op in id1.ops:
            for action in op.actions:
                ops.append(Op(op.entity, action))
        return (id1.storageId, ops)
    if not base64_decoded and _is_lower_case_hex_char(first):
        last = id.rfind(b'-')
        if last >= 0:
            storage_id = id[0:last]
    return (storage_id, [LOGIN_OP])