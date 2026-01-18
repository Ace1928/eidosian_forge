import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def validate_colors_dict(colors, colortype='tuple'):
    """
    Validates dictionary of color(s)
    """
    for key in colors:
        if 'rgb' in colors[key]:
            colors[key] = color_parser(colors[key], unlabel_rgb)
            for value in colors[key]:
                if value > 255.0:
                    raise exceptions.PlotlyError('Whoops! The elements in your rgb colors tuples cannot exceed 255.0.')
            colors[key] = color_parser(colors[key], unconvert_from_RGB_255)
        if '#' in colors[key]:
            colors[key] = color_parser(colors[key], hex_to_rgb)
            colors[key] = color_parser(colors[key], unconvert_from_RGB_255)
        if isinstance(colors[key], tuple):
            for value in colors[key]:
                if value > 1.0:
                    raise exceptions.PlotlyError('Whoops! The elements in your colors tuples cannot exceed 1.0.')
    if colortype == 'rgb':
        for key in colors:
            colors[key] = color_parser(colors[key], convert_to_RGB_255)
            colors[key] = color_parser(colors[key], label_rgb)
    return colors