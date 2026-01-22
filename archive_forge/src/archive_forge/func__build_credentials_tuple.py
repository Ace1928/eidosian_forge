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
def _build_credentials_tuple(mech: str, source: Optional[str], user: str, passwd: str, extra: Mapping[str, Any], database: Optional[str]) -> MongoCredential:
    """Build and return a mechanism specific credentials tuple."""
    if mech not in ('MONGODB-X509', 'MONGODB-AWS', 'MONGODB-OIDC') and user is None:
        raise ConfigurationError(f'{mech} requires a username.')
    if mech == 'GSSAPI':
        if source is not None and source != '$external':
            raise ValueError('authentication source must be $external or None for GSSAPI')
        properties = extra.get('authmechanismproperties', {})
        service_name = properties.get('SERVICE_NAME', 'mongodb')
        canonicalize = properties.get('CANONICALIZE_HOST_NAME', False)
        service_realm = properties.get('SERVICE_REALM')
        props = GSSAPIProperties(service_name=service_name, canonicalize_host_name=canonicalize, service_realm=service_realm)
        return MongoCredential(mech, '$external', user, passwd, props, None)
    elif mech == 'MONGODB-X509':
        if passwd is not None:
            raise ConfigurationError('Passwords are not supported by MONGODB-X509')
        if source is not None and source != '$external':
            raise ValueError('authentication source must be $external or None for MONGODB-X509')
        return MongoCredential(mech, '$external', user, None, None, None)
    elif mech == 'MONGODB-AWS':
        if user is not None and passwd is None:
            raise ConfigurationError('username without a password is not supported by MONGODB-AWS')
        if source is not None and source != '$external':
            raise ConfigurationError('authentication source must be $external or None for MONGODB-AWS')
        properties = extra.get('authmechanismproperties', {})
        aws_session_token = properties.get('AWS_SESSION_TOKEN')
        aws_props = _AWSProperties(aws_session_token=aws_session_token)
        return MongoCredential(mech, '$external', user, passwd, aws_props, None)
    elif mech == 'MONGODB-OIDC':
        properties = extra.get('authmechanismproperties', {})
        request_token_callback = properties.get('request_token_callback')
        provider_name = properties.get('PROVIDER_NAME', '')
        default_allowed = ['*.mongodb.net', '*.mongodb-dev.net', '*.mongodb-qa.net', '*.mongodbgov.net', 'localhost', '127.0.0.1', '::1']
        allowed_hosts = properties.get('allowed_hosts', default_allowed)
        if not request_token_callback and provider_name != 'aws':
            raise ConfigurationError("authentication with MONGODB-OIDC requires providing an request_token_callback or a provider_name of 'aws'")
        oidc_props = _OIDCProperties(request_token_callback=request_token_callback, provider_name=provider_name, allowed_hosts=allowed_hosts)
        return MongoCredential(mech, '$external', user, passwd, oidc_props, _Cache())
    elif mech == 'PLAIN':
        source_database = source or database or '$external'
        return MongoCredential(mech, source_database, user, passwd, None, None)
    else:
        source_database = source or database or 'admin'
        if passwd is None:
            raise ConfigurationError('A password is required.')
        return MongoCredential(mech, source_database, user, passwd, None, _Cache())