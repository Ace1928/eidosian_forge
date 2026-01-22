from __future__ import annotations
from typing import Final
from streamlit import util
from streamlit.logger import get_logger
def _make_blocking_http_get(url: str, timeout: float=5) -> str | None:
    import requests
    try:
        text = requests.get(url, timeout=timeout).text
        if isinstance(text, str):
            text = text.strip()
        return text
    except Exception:
        return None