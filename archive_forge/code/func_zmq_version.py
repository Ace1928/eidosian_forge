from __future__ import annotations
import re
from typing import Match, cast
from zmq.backend import zmq_version_info
def zmq_version() -> str:
    """return the version of libzmq as a string"""
    return '%i.%i.%i' % zmq_version_info()