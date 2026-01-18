import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def validate_colors(colors, colortype='tuple'):
    """
    Validates color(s) and returns a list of color(s) of a specified type
    """
    from numbers import Number
    if colors is None:
        colors = DEFAULT_PLOTLY_COLORS
    if isinstance(colors, str):
        if colors in PLOTLY_SCALES:
            colors_list = colorscale_to_colors(PLOTLY_SCALES[colors])
            colors = [colors_list[0]] + [colors_list[-1]]
        elif 'rgb' in colors or '#' in colors:
            colors = [colors]
        else:
            raise exceptions.PlotlyError('If your colors variable is a string, it must be a Plotly scale, an rgb color or a hex color.')
    elif isinstance(colors, tuple):
        if isinstance(colors[0], Number):
            colors = [colors]
        else:
            colors = list(colors)
    for j, each_color in enumerate(colors):
        if 'rgb' in each_color:
            each_color = color_parser(each_color, unlabel_rgb)
            for value in each_color:
                if value > 255.0:
                    raise exceptions.PlotlyError('Whoops! The elements in your rgb colors tuples cannot exceed 255.0.')
            each_color = color_parser(each_color, unconvert_from_RGB_255)
            colors[j] = each_color
        if '#' in each_color:
            each_color = color_parser(each_color, hex_to_rgb)
            each_color = color_parser(each_color, unconvert_from_RGB_255)
            colors[j] = each_color
        if isinstance(each_color, tuple):
            for value in each_color:
                if value > 1.0:
                    raise exceptions.PlotlyError('Whoops! The elements in your colors tuples cannot exceed 1.0.')
            colors[j] = each_color
    if colortype == 'rgb' and (not isinstance(colors, str)):
        for j, each_color in enumerate(colors):
            rgb_color = color_parser(each_color, convert_to_RGB_255)
            colors[j] = color_parser(rgb_color, label_rgb)
    return colors