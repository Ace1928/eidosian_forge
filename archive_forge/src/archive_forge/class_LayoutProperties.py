import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
@doc_subst(_doc_snippets)
class LayoutProperties(HasTraits):
    """Mixin class for layout templates

    This class handles mainly style attributes (height, grid_gap etc.)

    Parameters
    ----------

    {style_params}


    Note
    ----

    This class is only meant to be used in inheritance as mixin with other
    classes. It will not work, unless `self.layout` attribute is defined.

    """
    grid_gap = Unicode(None, allow_none=True, help='The grid-gap CSS attribute.')
    justify_content = CaselessStrEnum(['flex-start', 'flex-end', 'center', 'space-between', 'space-around'], allow_none=True, help='The justify-content CSS attribute.')
    align_items = CaselessStrEnum(['top', 'bottom', 'flex-start', 'flex-end', 'center', 'baseline', 'stretch'], allow_none=True, help='The align-items CSS attribute.')
    width = Unicode(None, allow_none=True, help='The width CSS attribute.')
    height = Unicode(None, allow_none=True, help='The width CSS attribute.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._property_rewrite = defaultdict(dict)
        self._property_rewrite['align_items'] = {'top': 'flex-start', 'bottom': 'flex-end'}
        self._copy_layout_props()
        self._set_observers()

    def _delegate_to_layout(self, change):
        """delegate the trait types to their counterparts in self.layout"""
        value, name = (change['new'], change['name'])
        value = self._property_rewrite[name].get(value, value)
        setattr(self.layout, name, value)

    def _set_observers(self):
        """set observers on all layout properties defined in this class"""
        _props = LayoutProperties.class_trait_names()
        self.observe(self._delegate_to_layout, _props)

    def _copy_layout_props(self):
        _props = LayoutProperties.class_trait_names()
        for prop in _props:
            value = getattr(self, prop)
            if value:
                value = self._property_rewrite[prop].get(value, value)
                setattr(self.layout, prop, value)