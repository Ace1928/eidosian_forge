from __future__ import annotations
import re
from streamlit import cli_util
from streamlit.config_option import ConfigOption
def append_desc(text):
    out.append('# ' + cli_util.style_for_cli(text, bold=True))