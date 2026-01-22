import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
@staticmethod
def _size_to_css(size):
    if re.match('\\d+\\.?\\d*(px|fr|%)$', size):
        return size
    if re.match('\\d+\\.?\\d*$', size):
        return size + 'fr'
    raise TypeError("the pane sizes must be in one of the following formats: '10px', '10fr', 10 (will be converted to '10fr').Got '{}'".format(size))