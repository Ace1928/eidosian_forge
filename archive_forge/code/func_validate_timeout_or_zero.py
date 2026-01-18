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
def validate_timeout_or_zero(option: str, value: Any) -> float:
    """Validates a timeout specified in milliseconds returning
    a value in floating point seconds for the case where None is an error
    and 0 is valid. Setting the timeout to nothing in the URI string is a
    config error.
    """
    if value is None:
        raise ConfigurationError(f'{option} cannot be None')
    if value == 0 or value == '0':
        return 0
    return validate_positive_float(option, value) / 1000.0