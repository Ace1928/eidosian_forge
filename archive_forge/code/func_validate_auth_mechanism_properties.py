from __future__ import annotations
import datetime
import inspect
import warnings
from collections import OrderedDict, abc
from typing import (
from urllib.parse import unquote_plus
from bson import SON
from bson.binary import UuidRepresentation
from bson.codec_options import CodecOptions, DatetimeConversion, TypeRegistry
from bson.raw_bson import RawBSONDocument
from pymongo.auth import MECHANISMS
from pymongo.compression_support import (
from pymongo.driver_info import DriverInfo
from pymongo.errors import ConfigurationError
from pymongo.monitoring import _validate_event_listeners
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import _MONGOS_MODES, _ServerMode
from pymongo.server_api import ServerApi
from pymongo.write_concern import DEFAULT_WRITE_CONCERN, WriteConcern, validate_boolean
def validate_auth_mechanism_properties(option: str, value: Any) -> dict[str, Union[bool, str]]:
    """Validate authMechanismProperties."""
    props: dict[str, Any] = {}
    if not isinstance(value, str):
        if not isinstance(value, dict):
            raise ValueError('Auth mechanism properties must be given as a string or a dictionary')
        for key, value in value.items():
            if isinstance(value, str):
                props[key] = value
            elif isinstance(value, bool):
                props[key] = str(value).lower()
            elif key in ['allowed_hosts'] and isinstance(value, list):
                props[key] = value
            elif inspect.isfunction(value):
                signature = inspect.signature(value)
                if key == 'request_token_callback':
                    expected_params = 2
                else:
                    raise ValueError(f'Unrecognized Auth mechanism function {key}')
                if len(signature.parameters) != expected_params:
                    msg = f'{key} must accept {expected_params} parameters'
                    raise ValueError(msg)
                props[key] = value
            else:
                raise ValueError('Auth mechanism property values must be strings or callback functions')
        return props
    value = validate_string(option, value)
    for opt in value.split(','):
        try:
            key, val = opt.split(':')
        except ValueError:
            if 'AWS_SESSION_TOKEN' in opt:
                opt = 'AWS_SESSION_TOKEN:<redacted token>, did you forget to percent-escape the token with quote_plus?'
            raise ValueError(f'auth mechanism properties must be key:value pairs like SERVICE_NAME:mongodb, not {opt}.') from None
        if key not in _MECHANISM_PROPS:
            raise ValueError(f'{key} is not a supported auth mechanism property. Must be one of {tuple(_MECHANISM_PROPS)}.')
        if key == 'CANONICALIZE_HOST_NAME':
            props[key] = validate_boolean_or_string(key, val)
        else:
            props[key] = unquote_plus(val)
    return props