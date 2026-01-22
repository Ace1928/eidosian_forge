from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget import register
from .widget_core import CoreWidget
from .trait_types import Date, date_serialization
from traitlets import Unicode, Bool, Union, CInt, CaselessStrEnum, TraitError, validate
@validate('min')
def _validate_min(self, proposal):
    """Enforce min <= value <= max"""
    min = proposal['value']
    if min is None:
        return min
    if self.max and min > self.max:
        raise TraitError('Setting min > max')
    if self.value and min > self.value:
        self.value = min
    return min