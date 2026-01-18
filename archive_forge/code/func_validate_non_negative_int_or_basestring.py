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
def validate_non_negative_int_or_basestring(option: Any, value: Any) -> Union[int, str]:
    """Validates that 'value' is an integer or string."""
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        try:
            val = int(value)
        except ValueError:
            return value
        return validate_non_negative_integer(option, val)
    raise TypeError(f'Wrong type for {option}, value must be an non negative integer or a string')