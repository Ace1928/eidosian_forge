from collections.abc import Iterable, Mapping
from itertools import chain
from .widget_description import DescriptionWidget, DescriptionStyle
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget_style import Style
from .trait_types import InstanceDict, TypedTuple
from .widget import register, widget_serialization
from .widget_int import SliderStyle
from .docutils import doc_subst
from traitlets import (Unicode, Bool, Int, Any, Dict, TraitError, CaselessStrEnum,
@register
@doc_subst(_doc_snippets)
class ToggleButtons(_Selection):
    """Group of toggle buttons that represent an enumeration.

    Only one toggle button can be toggled at any point in time.

    Parameters
    ----------
    {selection_params}

    tooltips: list
        Tooltip for each button. If specified, must be the
        same length as `options`.

    icons: list
        Icons to show on the buttons. This must be the name
        of a font-awesome icon. See `http://fontawesome.io/icons/`
        for a list of icons.

    button_style: str
        One of 'primary', 'success', 'info', 'warning' or
        'danger'. Applies a predefined style to every button.

    style: ToggleButtonsStyle
        Style parameters for the buttons.
    """
    _view_name = Unicode('ToggleButtonsView').tag(sync=True)
    _model_name = Unicode('ToggleButtonsModel').tag(sync=True)
    tooltips = TypedTuple(Unicode(), help='Tooltips for each button.').tag(sync=True)
    icons = TypedTuple(Unicode(), help='Icons names for each button (FontAwesome names without the fa- prefix).').tag(sync=True)
    style = InstanceDict(ToggleButtonsStyle).tag(sync=True, **widget_serialization)
    button_style = CaselessStrEnum(values=['primary', 'success', 'info', 'warning', 'danger', ''], default_value='', allow_none=True, help='Use a predefined styling for the buttons.').tag(sync=True)