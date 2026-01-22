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
def _macaroon_id_ops(ops):
    """Return operations suitable for serializing as part of a MacaroonId.

    It assumes that ops has been canonicalized and that there's at least
    one operation.
    """
    id_ops = []
    for entity, entity_ops in itertools.groupby(ops, lambda x: x.entity):
        actions = map(lambda x: x.action, entity_ops)
        id_ops.append(id_pb2.Op(entity=entity, actions=actions))
    return id_ops