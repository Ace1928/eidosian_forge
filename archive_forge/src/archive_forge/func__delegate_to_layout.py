import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
def _delegate_to_layout(self, change):
    """delegate the trait types to their counterparts in self.layout"""
    value, name = (change['new'], change['name'])
    value = self._property_rewrite[name].get(value, value)
    setattr(self.layout, name, value)