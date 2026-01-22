from traitlets import Unicode, Bool, validate, TraitError
from .trait_types import datetime_serialization, Datetime, naive_serialization
from .valuewidget import ValueWidget
from .widget import register
from .widget_core import CoreWidget
from .widget_description import DescriptionWidget
def _validate_tz(self, value):
    if value.tzinfo is not None:
        raise TraitError('%s values needs to be timezone unaware' % (self.__class__.__name__,))
    return value