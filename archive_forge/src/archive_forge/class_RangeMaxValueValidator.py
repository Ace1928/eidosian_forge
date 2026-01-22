from django.core.exceptions import ValidationError
from django.core.validators import (
from django.utils.deconstruct import deconstructible
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
class RangeMaxValueValidator(MaxValueValidator):

    def compare(self, a, b):
        return a.upper is None or a.upper > b
    message = _('Ensure that the upper bound of the range is not greater than %(limit_value)s.')