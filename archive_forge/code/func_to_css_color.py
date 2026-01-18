from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
def to_css_color(color: MaybeColor) -> Color:
    """Convert input into a CSS-compatible color that Vega can use.

    Inputs must be a hex string, rgb()/rgba() string, or a color tuple. Inputs may not be a CSS
    color name, other CSS color function (like "hsl(...)"), etc.

    See tests for more info.
    """
    if is_css_color_like(color):
        return cast(Color, color)
    if is_color_tuple_like(color):
        ctuple = cast(ColorTuple, color)
        ctuple = _normalize_tuple(ctuple, _int_formatter, _float_formatter)
        if len(ctuple) == 3:
            return f'rgb({ctuple[0]}, {ctuple[1]}, {ctuple[2]})'
        elif len(ctuple) == 4:
            c4tuple = cast(MixedRGBAColorTuple, ctuple)
            return f'rgba({c4tuple[0]}, {c4tuple[1]}, {c4tuple[2]}, {c4tuple[3]})'
    raise InvalidColorException(color)