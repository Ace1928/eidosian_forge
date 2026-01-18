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
@test.command('prog_name')
def test_prog_name():
    """Assert that the program name is set to `streamlit test`.

    This is used by our cli-smoke-tests to verify that the program name is set
    to `streamlit ...` whether the streamlit binary is invoked directly or via
    `python -m streamlit ...`.
    """
    _get_command_line_as_string()
    parent = click.get_current_context().parent
    assert parent is not None
    assert parent.command_path == 'streamlit test'