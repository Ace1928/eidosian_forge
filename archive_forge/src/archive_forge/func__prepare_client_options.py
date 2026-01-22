from __future__ import annotations
import datetime
from queue import Empty
import pymongo
from pymongo import MongoClient, errors, uri_parser
from pymongo.cursor import CursorType
from kombu.exceptions import VersionMismatch
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import bytes_to_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url
from . import virtual
from .base import to_rabbitmq_queue_arguments
def _prepare_client_options(self, options):
    if pymongo.version_tuple >= (3,):
        options.pop('auto_start_request', None)
        if isinstance(options.get('readpreference'), int):
            modes = pymongo.read_preferences._MONGOS_MODES
            options['readpreference'] = modes[options['readpreference']]
    return options