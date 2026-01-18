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
@property
def version(self):
    return self._version