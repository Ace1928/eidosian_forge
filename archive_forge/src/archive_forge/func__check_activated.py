from __future__ import annotations
import json
import os
import sys
import textwrap
from collections import namedtuple
from datetime import datetime
from typing import Final, NoReturn
from uuid import uuid4
from streamlit import cli_util, env_util, file_util, util
from streamlit.logger import get_logger
def _check_activated(self, auto_resolve: bool=True) -> None:
    """Check if streamlit is activated.

        Used by `streamlit run script.py`
        """
    try:
        self.load(auto_resolve)
    except (Exception, RuntimeError) as e:
        _exit(str(e))
    if self.activation is None or not self.activation.is_valid:
        _exit('Activation email not valid.')