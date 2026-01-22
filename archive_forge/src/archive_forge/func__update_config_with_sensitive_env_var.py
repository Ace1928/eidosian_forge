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
def _update_config_with_sensitive_env_var(config_options: dict[str, ConfigOption]):
    """Update the config system by parsing the environment variable.

    This should only be called from get_config_options.
    """
    for opt_name, opt_val in config_options.items():
        if not opt_val.sensitive:
            continue
        env_var_value = os.environ.get(opt_val.env_var)
        if env_var_value is None:
            continue
        _set_option(opt_name, env_var_value, _DEFINED_BY_ENV_VAR)