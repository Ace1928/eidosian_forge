from .widget_description import DescriptionStyle, DescriptionWidget
from .widget_core import CoreWidget
from .valuewidget import ValueWidget
from .widget import register, widget_serialization
from .trait_types import Color, InstanceDict
from traitlets import Unicode, Bool, CaselessStrEnum
Displays a boolean `value` in the form of a green check (True / valid)
    or a red cross (False / invalid).

    Parameters
    ----------
    value: {True,False}
        value of the Valid widget
    