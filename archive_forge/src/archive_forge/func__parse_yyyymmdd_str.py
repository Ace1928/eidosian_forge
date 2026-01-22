from __future__ import annotations
import datetime
import re
import textwrap
from typing import Any, Callable
from streamlit import util
from streamlit.case_converters import to_snake_case
def _parse_yyyymmdd_str(date_str: str) -> datetime.datetime:
    year, month, day = (int(token) for token in date_str.split('-', 2))
    return datetime.datetime(year, month, day)