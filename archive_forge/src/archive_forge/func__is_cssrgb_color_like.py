from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
def _is_cssrgb_color_like(color: MaybeColor) -> bool:
    """Check whether the input looks like a CSS rgb() or rgba() color string.

    This is meant to be lightweight, and not a definitive answer. The definitive solution is to try
    to convert and see if an error is thrown.

    NOTE: We only accept hex colors and color tuples as user input. So do not use this function to
    validate user input! Instead use is_hex_color_like and is_color_tuple_like.
    """
    return isinstance(color, str) and (color.startswith('rgb(') or color.startswith('rgba('))