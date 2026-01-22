from __future__ import annotations
import re
from streamlit import cli_util
from streamlit.config_option import ConfigOption
def _clean_paragraphs(txt: str) -> list[str]:
    """Split the text into paragraphs, preserve newlines within the paragraphs."""
    txt = txt.strip('\n')
    paragraphs = txt.split('\n\n')
    cleaned_paragraphs = ['\n'.join((_clean(line) for line in paragraph.split('\n'))) for paragraph in paragraphs]
    return cleaned_paragraphs