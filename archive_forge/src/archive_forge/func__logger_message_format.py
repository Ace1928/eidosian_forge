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
@_create_option('logger.messageFormat', type_=str)
def _logger_message_format() -> str:
    """String format for logging messages. If logger.datetimeFormat is set,
    logger messages will default to `%(asctime)s.%(msecs)03d %(message)s`. See
    [Python's documentation](https://docs.python.org/2.6/library/logging.html#formatter-objects)
    for available attributes.

    Default: "%(asctime)s %(message)s"
    """
    if get_option('global.developmentMode'):
        from streamlit.logger import DEFAULT_LOG_MESSAGE
        return DEFAULT_LOG_MESSAGE
    else:
        return '%(asctime)s %(message)s'