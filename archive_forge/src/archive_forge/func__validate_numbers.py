from traitlets import (
from .widget_description import DescriptionWidget
from .valuewidget import ValueWidget
from .widget_core import CoreWidget
from .widget import register
from .trait_types import Color, NumberFormat
@validate('value')
def _validate_numbers(self, proposal):
    for tag_value in proposal['value']:
        if self.min is not None and tag_value < self.min:
            raise TraitError('Tag value {} should be >= {}'.format(tag_value, self.min))
        if self.max is not None and tag_value > self.max:
            raise TraitError('Tag value {} should be <= {}'.format(tag_value, self.max))
    return proposal['value']