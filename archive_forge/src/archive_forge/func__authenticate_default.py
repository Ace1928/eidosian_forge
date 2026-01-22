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
def _authenticate_default(credentials: MongoCredential, conn: Connection) -> None:
    if conn.max_wire_version >= 7:
        if conn.negotiated_mechs:
            mechs = conn.negotiated_mechs
        else:
            source = credentials.source
            cmd = conn.hello_cmd()
            cmd['saslSupportedMechs'] = source + '.' + credentials.username
            mechs = conn.command(source, cmd, publish_events=False).get('saslSupportedMechs', [])
        if 'SCRAM-SHA-256' in mechs:
            return _authenticate_scram(credentials, conn, 'SCRAM-SHA-256')
        else:
            return _authenticate_scram(credentials, conn, 'SCRAM-SHA-1')
    else:
        return _authenticate_scram(credentials, conn, 'SCRAM-SHA-1')