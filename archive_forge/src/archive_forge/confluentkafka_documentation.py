from __future__ import annotations
from queue import Empty
from kombu.transport import virtual
from kombu.utils import cached_property
from kombu.utils.encoding import str_to_bytes
from kombu.utils.json import dumps, loads
from kombu.log import get_logger
Check if a topic already exists.