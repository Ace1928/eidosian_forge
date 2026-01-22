from __future__ import annotations
import os
import sys
from typing import Any
import click
import streamlit.runtime.caching as caching
import streamlit.runtime.legacy_caching as legacy_caching
import streamlit.web.bootstrap as bootstrap
from streamlit import config as _config
from streamlit.config_option import ConfigOption
from streamlit.runtime.credentials import Credentials, check_credentials
from streamlit.web.cache_storage_manager_config import (
def _make_sensitive_option_callback(config_option: ConfigOption):

    def callback(_ctx: click.Context, _param: click.Parameter, cli_value) -> None:
        if cli_value is None:
            return None
        raise SystemExit(f'Setting {config_option.key!r} option using the CLI flag is not allowed. Set this option in the configuration file or environment variable: {config_option.env_var!r}')
    return callback