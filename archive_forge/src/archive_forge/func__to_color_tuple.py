from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
def _to_color_tuple(color: MaybeColor, rgb_formatter: Callable[[float, MaybeColor], float], alpha_formatter: Callable[[float, MaybeColor], float]):
    """Convert a potential color to a color tuple.

    The exact type of color tuple this outputs is dictated by the formatter parameters.

    The R, G, B components are transformed by rgb_formatter, and the alpha component is transformed
    by alpha_formatter.

    For example, to output a (float, float, float, int) color tuple, set rgb_formatter
    to _float_formatter and alpha_formatter to _int_formatter.
    """
    if is_hex_color_like(color):
        hex_len = len(color)
        color_hex = cast(str, color)
        if hex_len == 4:
            r = 2 * color_hex[1]
            g = 2 * color_hex[2]
            b = 2 * color_hex[3]
            a = 'ff'
        elif hex_len == 5:
            r = 2 * color_hex[1]
            g = 2 * color_hex[2]
            b = 2 * color_hex[3]
            a = 2 * color_hex[4]
        elif hex_len == 7:
            r = color_hex[1:3]
            g = color_hex[3:5]
            b = color_hex[5:7]
            a = 'ff'
        elif hex_len == 9:
            r = color_hex[1:3]
            g = color_hex[3:5]
            b = color_hex[5:7]
            a = color_hex[7:9]
        else:
            raise InvalidColorException(color)
        try:
            color = (int(r, 16), int(g, 16), int(b, 16), int(a, 16))
        except:
            raise InvalidColorException(color)
    if is_color_tuple_like(color):
        color_tuple = cast(ColorTuple, color)
        return _normalize_tuple(color_tuple, rgb_formatter, alpha_formatter)
    raise InvalidColorException(color)