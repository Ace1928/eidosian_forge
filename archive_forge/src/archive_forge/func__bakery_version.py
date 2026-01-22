import abc
import base64
import json
import logging
import os
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery._utils import b64decode
from pymacaroons.serializers import json_serializer
from ._versions import (
from ._error import (
from ._codec import (
from ._keys import PublicKey
from ._third_party import (
def _bakery_version(v):
    if v == pymacaroons.MACAROON_V1:
        return VERSION_1
    elif v == pymacaroons.MACAROON_V2:
        return VERSION_2
    else:
        raise ValueError('unknown macaroon version when deserializing legacy bakery macaroon; got {}'.format(v))