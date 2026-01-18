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
def validate_read_preference_tags(name: str, value: Any) -> list[dict[str, str]]:
    """Parse readPreferenceTags if passed as a client kwarg."""
    if not isinstance(value, list):
        value = [value]
    tag_sets: list = []
    for tag_set in value:
        if tag_set == '':
            tag_sets.append({})
            continue
        try:
            tags = {}
            for tag in tag_set.split(','):
                key, val = tag.split(':')
                tags[unquote_plus(key)] = unquote_plus(val)
            tag_sets.append(tags)
        except Exception:
            raise ValueError(f'{tag_set!r} not a valid value for {name}') from None
    return tag_sets