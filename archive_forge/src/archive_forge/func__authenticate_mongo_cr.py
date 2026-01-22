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
def _authenticate_mongo_cr(credentials: MongoCredential, conn: Connection) -> None:
    """Authenticate using MONGODB-CR."""
    source = credentials.source
    username = credentials.username
    password = credentials.password
    response = conn.command(source, {'getnonce': 1})
    nonce = response['nonce']
    key = _auth_key(nonce, username, password)
    query = SON([('authenticate', 1), ('user', username), ('nonce', nonce), ('key', key)])
    conn.command(source, query)