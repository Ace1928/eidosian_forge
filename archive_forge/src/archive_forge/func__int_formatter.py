from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
def _int_formatter(component: float, color: MaybeColor) -> int:
    """Convert a color component (float or int) to an int from 0 to 255.

    Anything too small will become 0, and anything too large will become 255.
    """
    if isinstance(component, float):
        component = int(component * 255)
    if isinstance(component, int):
        return min(255, max(component, 0))
    raise InvalidColorException(color)