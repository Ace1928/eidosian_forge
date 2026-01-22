from __future__ import annotations
import logging # isort:skip
from math import inf
from ...core.enums import (
from ...core.properties import (
from ...core.property_aliases import BorderRadius
from ...core.property_mixins import (
from ..common.properties import Coordinate
from ..nodes import BoxNodes, Node
from .annotation import Annotation, DataAnnotation
from .arrows import ArrowHead, TeeHead
class BoxAnnotation(Annotation):
    """ Render a shaded rectangular region as an annotation.

    See :ref:`ug_basic_annotations_box_annotations` for information on plotting box annotations.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    left = Coordinate(default=lambda: Node.frame.left, help='\n    The x-coordinates of the left edge of the box annotation.\n    ').accepts(Null, lambda _: Node.frame.left)
    right = Coordinate(default=lambda: Node.frame.right, help='\n    The x-coordinates of the right edge of the box annotation.\n    ').accepts(Null, lambda _: Node.frame.right)
    top = Coordinate(default=lambda: Node.frame.top, help='\n    The y-coordinates of the top edge of the box annotation.\n    ').accepts(Null, lambda _: Node.frame.top)
    bottom = Coordinate(default=lambda: Node.frame.bottom, help='\n    The y-coordinates of the bottom edge of the box annotation.\n    ').accepts(Null, lambda _: Node.frame.bottom)
    left_units = Enum(CoordinateUnits, default='data', help="\n    The unit type for the left attribute. Interpreted as |data units| by\n    default. This doesn't have any effect if ``left`` is a node.\n    ")
    right_units = Enum(CoordinateUnits, default='data', help="\n    The unit type for the right attribute. Interpreted as |data units| by\n    default. This doesn't have any effect if ``right`` is a node.\n    ")
    top_units = Enum(CoordinateUnits, default='data', help="\n    The unit type for the top attribute. Interpreted as |data units| by\n    default. This doesn't have any effect if ``top`` is a node.\n    ")
    bottom_units = Enum(CoordinateUnits, default='data', help="\n    The unit type for the bottom attribute. Interpreted as |data units| by\n    default. This doesn't have any effect if ``bottom`` is a node.\n    ")
    left_limit = Nullable(Coordinate, help='\n    Optional left limit for box movement.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    right_limit = Nullable(Coordinate, help='\n    Optional right limit for box movement.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    top_limit = Nullable(Coordinate, help='\n    Optional top limit for box movement.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    bottom_limit = Nullable(Coordinate, help='\n    Optional bottom limit for box movement.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    min_width = NonNegative(Float, default=0, help='\n    Allows to set the minium width of the box.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    min_height = NonNegative(Float, default=0, help='\n    Allows to set the maximum width of the box.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    max_width = Positive(Float, default=inf, help='\n    Allows to set the minium height of the box.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    max_height = Positive(Float, default=inf, help='\n    Allows to set the maximum height of the box.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    border_radius = BorderRadius(default=0, help='\n    Allows the box to have rounded corners.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    editable = Bool(default=False, help='\n    Allows to interactively modify the geometry of this box.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    resizable = Enum(Resizable, default='all', help='\n    If `editable` is set, this property allows to configure which\n    combinations of edges are allowed to be moved, thus allows\n    restrictions on resizing of the box.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    movable = Enum(Movable, default='both', help='\n    If `editable` is set, this property allows to configure in which\n    directions the box can be moved.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    symmetric = Bool(default=False, help='\n    Indicates whether the box is resizable around its center or its corners.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    line_props = Include(ScalarLineProps, help='\n    The {prop} values for the box.\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the box.\n    ')
    hatch_props = Include(ScalarHatchProps, help='\n    The {prop} values for the box.\n    ')
    hover_line_props = Include(ScalarLineProps, prefix='hover', help='\n    The {prop} values for the box when hovering over.\n    ')
    hover_fill_props = Include(ScalarFillProps, prefix='hover', help='\n    The {prop} values for the box when hovering over.\n    ')
    hover_hatch_props = Include(ScalarHatchProps, prefix='hover', help='\n    The {prop} values for the box when hovering over.\n    ')
    line_color = Override(default='#cccccc')
    line_alpha = Override(default=0.3)
    fill_color = Override(default='#fff9ba')
    fill_alpha = Override(default=0.4)
    hover_line_color = Override(default=None)
    hover_line_alpha = Override(default=0.3)
    hover_fill_color = Override(default=None)
    hover_fill_alpha = Override(default=0.4)

    @property
    def nodes(self) -> BoxNodes:
        return BoxNodes(self)