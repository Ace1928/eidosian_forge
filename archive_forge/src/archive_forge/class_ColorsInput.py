from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget import register
from .trait_types import Color, NumberFormat
@register
class ColorsInput(TagsInputBase):
    """
    List of color tags
    """
    _model_name = Unicode('ColorsInputModel').tag(sync=True)
    _view_name = Unicode('ColorsInputView').tag(sync=True)
    value = List(Color(), help='List of string tags').tag(sync=True)