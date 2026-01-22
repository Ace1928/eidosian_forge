from __future__ import annotations
import copy
import os
import secrets
import threading
from collections import OrderedDict
from typing import Any, Callable
from blinker import Signal
from streamlit import config_util, development, env_util, file_util, util
from streamlit.config_option import ConfigOption
from streamlit.errors import StreamlitAPIException
@_create_option('server.address')
def _server_address() -> str | None:
    """The address where the server will listen for client and browser
    connections. Use this if you want to bind the server to a specific address.
    If set, the server will only be accessible from this address, and not from
    any aliases (like localhost).

    Default: (unset)
    """
    return None