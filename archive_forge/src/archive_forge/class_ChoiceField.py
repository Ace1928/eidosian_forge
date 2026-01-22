import copy
import datetime
import json
import math
import operator
import os
import re
import uuid
import warnings
from decimal import Decimal, DecimalException
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit
from django.conf import settings
from django.core import validators
from django.core.exceptions import ValidationError
from django.forms.boundfield import BoundField
from django.forms.utils import from_current_timezone, to_current_timezone
from django.forms.widgets import (
from django.utils import formats
from django.utils.choices import normalize_choices
from django.utils.dateparse import parse_datetime, parse_duration
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.duration import duration_string
from django.utils.ipv6 import clean_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
class ChoiceField(Field):
    widget = Select
    default_error_messages = {'invalid_choice': _('Select a valid choice. %(value)s is not one of the available choices.')}

    def __init__(self, *, choices=(), **kwargs):
        super().__init__(**kwargs)
        self.choices = choices

    def __deepcopy__(self, memo):
        result = super().__deepcopy__(memo)
        result._choices = copy.deepcopy(self._choices, memo)
        return result

    @property
    def choices(self):
        return self._choices

    @choices.setter
    def choices(self, value):
        self._choices = self.widget.choices = normalize_choices(value)

    def to_python(self, value):
        """Return a string."""
        if value in self.empty_values:
            return ''
        return str(value)

    def validate(self, value):
        """Validate that the input is in self.choices."""
        super().validate(value)
        if value and (not self.valid_value(value)):
            raise ValidationError(self.error_messages['invalid_choice'], code='invalid_choice', params={'value': value})

    def valid_value(self, value):
        """Check to see if the provided value is a valid choice."""
        text_value = str(value)
        for k, v in self.choices:
            if isinstance(v, (list, tuple)):
                for k2, v2 in v:
                    if value == k2 or text_value == str(k2):
                        return True
            elif value == k or text_value == str(k):
                return True
        return False