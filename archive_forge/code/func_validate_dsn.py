import os
import pathlib
import typing
import threading
import functools
import hashlib
from .compat import validator, root_validator, Field, PYD_VERSION, get_pyd_field_names, get_pyd_dict, pyd_parse_obj, get_pyd_schema
from .compat import BaseSettings as _BaseSettings
from .compat import BaseModel as _BaseModel
from pydantic.networks import AnyUrl, Url, MultiHostUrl
@validator('dsn')
def validate_dsn(cls, v):
    if isinstance(v, str):
        v = KeyDBDsn(v)
    if v.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f'Invalid scheme {v.scheme} for KeyDBUri')
    return v