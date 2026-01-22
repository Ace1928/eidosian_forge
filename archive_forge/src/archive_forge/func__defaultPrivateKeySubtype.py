from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage
def _defaultPrivateKeySubtype(keyType):
    """
    Return a reasonable default private key subtype for a given key type.

    @type keyType: L{str}
    @param keyType: A key type, as returned by
        L{twisted.conch.ssh.keys.Key.type}.

    @rtype: L{str}
    @return: A private OpenSSH key subtype (C{'PEM'} or C{'v1'}).
    """
    if keyType == 'Ed25519':
        return 'v1'
    else:
        return 'PEM'