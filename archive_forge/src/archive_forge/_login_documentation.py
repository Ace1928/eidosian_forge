import os
import subprocess
from functools import partial
from getpass import getpass
from pathlib import Path
from typing import Optional
from . import constants
from .commands._cli_utils import ANSI
from .utils import (
from .utils._token import _get_token_from_environment, _get_token_from_google_colab

        Event handler for the login button.

        Args:
            write_permission (`bool`, defaults to `False`):
                If `True`, requires a token with write permission.
        