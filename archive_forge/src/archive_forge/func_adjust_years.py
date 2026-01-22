from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Literal, overload
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
def adjust_years(input_date: date, years: int) -> date:
    """Add or subtract years from a date."""
    try:
        return input_date.replace(year=input_date.year + years)
    except ValueError as err:
        if input_date.month == 2 and input_date.day == 29:
            return input_date.replace(year=input_date.year + years, month=2, day=28)
        raise StreamlitAPIException(f'Date {input_date} does not exist in the target year {input_date.year + years}. This should never happen. Please report this bug.') from err