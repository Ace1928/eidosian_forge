import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def _axis_spanning_shapes_docstr(shape_type):
    docstr = ''
    if shape_type == 'hline':
        docstr = '\nAdd a horizontal line to a plot or subplot that extends infinitely in the\nx-dimension.\n\nParameters\n----------\ny: float or int\n    A number representing the y coordinate of the horizontal line.'
    elif shape_type == 'vline':
        docstr = '\nAdd a vertical line to a plot or subplot that extends infinitely in the\ny-dimension.\n\nParameters\n----------\nx: float or int\n    A number representing the x coordinate of the vertical line.'
    elif shape_type == 'hrect':
        docstr = '\nAdd a rectangle to a plot or subplot that extends infinitely in the\nx-dimension.\n\nParameters\n----------\ny0: float or int\n    A number representing the y coordinate of one side of the rectangle.\ny1: float or int\n    A number representing the y coordinate of the other side of the rectangle.'
    elif shape_type == 'vrect':
        docstr = '\nAdd a rectangle to a plot or subplot that extends infinitely in the\ny-dimension.\n\nParameters\n----------\nx0: float or int\n    A number representing the x coordinate of one side of the rectangle.\nx1: float or int\n    A number representing the x coordinate of the other side of the rectangle.'
    docstr += '\nexclude_empty_subplots: Boolean\n    If True (default) do not place the shape on subplots that have no data\n    plotted on them.\nrow: None, int or \'all\'\n    Subplot row for shape indexed starting at 1. If \'all\', addresses all rows in\n    the specified column(s). If both row and col are None, addresses the\n    first subplot if subplots exist, or the only plot. By default is "all".\ncol: None, int or \'all\'\n    Subplot column for shape indexed starting at 1. If \'all\', addresses all rows in\n    the specified column(s). If both row and col are None, addresses the\n    first subplot if subplots exist, or the only plot. By default is "all".\nannotation: dict or plotly.graph_objects.layout.Annotation. If dict(),\n    it is interpreted as describing an annotation. The annotation is\n    placed relative to the shape based on annotation_position (see\n    below) unless its x or y value has been specified for the annotation\n    passed here. xref and yref are always the same as for the added\n    shape and cannot be overridden.'
    if shape_type in ['hline', 'vline']:
        docstr += '\nannotation_position: a string containing optionally ["top", "bottom"]\n    and ["left", "right"] specifying where the text should be anchored\n    to on the line. Example positions are "bottom left", "right top",\n    "right", "bottom". If an annotation is added but annotation_position is\n    not specified, this defaults to "top right".'
    elif shape_type in ['hrect', 'vrect']:
        docstr += '\nannotation_position: a string containing optionally ["inside", "outside"], ["top", "bottom"]\n    and ["left", "right"] specifying where the text should be anchored\n    to on the rectangle. Example positions are "outside top left", "inside\n    bottom", "right", "inside left", "inside" ("outside" is not supported). If\n    an annotation is added but annotation_position is not specified this\n    defaults to "inside top right".'
    docstr += '\nannotation_*: any parameters to go.layout.Annotation can be passed as\n    keywords by prefixing them with "annotation_". For example, to specify the\n    annotation text "example" you can pass annotation_text="example" as a\n    keyword argument.\n**kwargs:\n    Any named function parameters that can be passed to \'add_shape\',\n    except for x0, x1, y0, y1 or type.'
    return docstr