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
def add_caveat(self, cav, key=None, loc=None):
    """Add a caveat to the macaroon.

        It encrypts it using the given key pair
        and by looking up the location using the given locator.
        As a special case, if the caveat's Location field has the prefix
        "local " the caveat is added as a client self-discharge caveat using
        the public key base64-encoded in the rest of the location. In this
        case, the Condition field must be empty. The resulting third-party
        caveat will encode the condition "true" encrypted with that public
        key.

        @param cav the checkers.Caveat to be added.
        @param key the public key to encrypt third party caveat.
        @param loc locator to find information on third parties when adding
        third party caveats. It is expected to have a third_party_info method
        that will be called with a location string and should return a
        ThirdPartyInfo instance holding the requested information.
        """
    if cav.location is None:
        self._macaroon.add_first_party_caveat(self.namespace.resolve_caveat(cav).condition)
        return
    if key is None:
        raise ValueError('no private key to encrypt third party caveat')
    local_info = _parse_local_location(cav.location)
    if local_info is not None:
        if cav.condition:
            raise ValueError('cannot specify caveat condition in local third-party caveat')
        info = local_info
        cav = checkers.Caveat(location='local', condition='true')
    else:
        if loc is None:
            raise ValueError('no locator when adding third party caveat')
        info = loc.third_party_info(cav.location)
    root_key = os.urandom(24)
    if self._version < info.version:
        info = ThirdPartyInfo(version=self._version, public_key=info.public_key)
    caveat_info = encode_caveat(cav.condition, root_key, info, key, self._namespace)
    if info.version < VERSION_3:
        id = caveat_info
    else:
        id = self._new_caveat_id(self._caveat_id_prefix)
        self._caveat_data[id] = caveat_info
    self._macaroon.add_third_party_caveat(cav.location, root_key, id)