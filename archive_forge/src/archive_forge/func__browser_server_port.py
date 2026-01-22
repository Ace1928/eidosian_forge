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
@_create_option('browser.serverPort', type_=int)
def _browser_server_port() -> int:
    """Port where users should point their browsers in order to connect to the
    app.

    This is used to:
    - Set the correct URL for XSRF protection purposes.
    - Show the URL on the terminal (part of `streamlit run`).
    - Open the browser automatically (part of `streamlit run`).

    This option is for advanced use cases. To change the port of your app, use
    `server.Port` instead. Don't use port 3000 which is reserved for internal
    development.

    Default: whatever value is set in server.port.
    """
    return int(get_option('server.port'))