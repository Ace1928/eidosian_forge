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
class _MultipleSelectionNonempty(_MultipleSelection):
    """Selection that is guaranteed to have an option available."""

    def __init__(self, *args, **kwargs):
        if len(kwargs.get('options', ())) == 0:
            raise TraitError('options must be nonempty')
        super().__init__(*args, **kwargs)

    @validate('options')
    def _validate_options(self, proposal):
        proposal.value = _exhaust_iterable(proposal.value)
        self._options_full = _make_options(proposal.value)
        if len(self._options_full) == 0:
            raise TraitError('Option list must be nonempty')
        return proposal.value