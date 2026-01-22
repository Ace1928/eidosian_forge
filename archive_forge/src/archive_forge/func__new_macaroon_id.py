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
def _new_macaroon_id(self, storage_id, expiry, ops):
    nonce = os.urandom(16)
    if len(ops) == 1 or self.ops_store is None:
        return id_pb2.MacaroonId(nonce=nonce, storageId=storage_id, ops=_macaroon_id_ops(ops))
    entity = self.ops_entity(ops)
    self.ops_store.put_ops(entity, expiry, ops)
    return id_pb2.MacaroonId(nonce=nonce, storageId=storage_id, ops=[id_pb2.Op(entity=entity, actions=['*'])])