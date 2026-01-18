import copy
import re
import threading
import time
import warnings
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union
from redis._parsers.encoders import Encoder
from redis._parsers.helpers import (
from redis.commands import (
from redis.connection import (
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def try_read():
    if not block:
        if not conn.can_read(timeout=timeout):
            return None
    else:
        conn.connect()
    return conn.read_response(disconnect_on_error=False, push_request=True)