import collections.abc
import inspect
import warnings
from math import ceil
from django.utils.functional import cached_property
from django.utils.inspect import method_has_no_args
from django.utils.translation import gettext_lazy as _
def validate_number(self, number):
    """Validate the given 1-based page number."""
    try:
        if isinstance(number, float) and (not number.is_integer()):
            raise ValueError
        number = int(number)
    except (TypeError, ValueError):
        raise PageNotAnInteger(self.error_messages['invalid_page'])
    if number < 1:
        raise EmptyPage(self.error_messages['min_page'])
    if number > self.num_pages:
        raise EmptyPage(self.error_messages['no_results'])
    return number