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
def third_party_caveats(self):
    """Return the third party caveats.

        @return the third party caveats as pymacaroons caveats.
        """
    return self._macaroon.third_party_caveats()