from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
def _get_number_value(self, values):
    try:
        return values[number]
    except KeyError:
        raise KeyError("Your dictionary lacks key '%s'. Please provide it, because it is required to determine whether string is singular or plural." % number)