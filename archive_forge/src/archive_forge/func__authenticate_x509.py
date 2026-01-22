from __future__ import annotations
import functools
import hashlib
import hmac
import os
import socket
import typing
from base64 import standard_b64decode, standard_b64encode
from collections import namedtuple
from typing import (
from urllib.parse import quote
from bson.binary import Binary
from bson.son import SON
from pymongo.auth_aws import _authenticate_aws
from pymongo.auth_oidc import _authenticate_oidc, _get_authenticator, _OIDCProperties
from pymongo.errors import ConfigurationError, OperationFailure
from pymongo.saslprep import saslprep
def _authenticate_x509(credentials: MongoCredential, conn: Connection) -> None:
    """Authenticate using MONGODB-X509."""
    ctx = conn.auth_ctx
    if ctx and ctx.speculate_succeeded():
        return
    cmd = _X509Context(credentials, conn.address).speculate_command()
    conn.command('$external', cmd)