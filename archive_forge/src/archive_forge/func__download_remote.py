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
def _download_remote(main_script_path: str, url_path: str) -> None:
    """Fetch remote file at url_path to main_script_path"""
    import requests
    with open(main_script_path, 'wb') as fp:
        try:
            resp = requests.get(url_path)
            resp.raise_for_status()
            fp.write(resp.content)
        except requests.exceptions.RequestException as e:
            raise click.BadParameter(f'Unable to fetch {url_path}.\n{e}')