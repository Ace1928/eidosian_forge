from __future__ import annotations
import os
import socket
import string
from queue import Empty
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
Delete all current messages in a queue.