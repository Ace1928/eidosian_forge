from __future__ import annotations
import string
from queue import Empty
from typing import Any, Optional
from azure.core.exceptions import ResourceExistsError
from kombu.utils.encoding import safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
Delete all current messages in a queue.