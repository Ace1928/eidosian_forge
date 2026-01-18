from __future__ import annotations
import binascii
import json
import warnings
from typing import TYPE_CHECKING, Any
from .algorithms import (
from .exceptions import (
from .utils import base64url_decode, base64url_encode
from .warnings import RemovedInPyjwt3Warning
def unregister_algorithm(self, alg_id: str) -> None:
    """
        Unregisters an Algorithm for use when creating and verifying tokens
        Throws KeyError if algorithm is not registered.
        """
    if alg_id not in self._algorithms:
        raise KeyError('The specified algorithm could not be removed because it is not registered.')
    del self._algorithms[alg_id]
    self._valid_algs.remove(alg_id)