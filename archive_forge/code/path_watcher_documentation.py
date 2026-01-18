from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
Return the PathWatcher class that corresponds to the given watcher_type
    string. Acceptable values are 'auto', 'watchdog', 'poll' and 'none'.
    