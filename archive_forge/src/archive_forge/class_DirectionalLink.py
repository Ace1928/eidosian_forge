from .widget import Widget, register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Unicode, Tuple, Instance, TraitError
@register
class DirectionalLink(Link):
    """A directional link

    source: a (Widget, 'trait_name') tuple for the source trait
    target: a (Widget, 'trait_name') tuple that should be updated
    when the source trait changes.
    """
    _model_name = Unicode('DirectionalLinkModel').tag(sync=True)