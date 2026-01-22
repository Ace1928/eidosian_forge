import copy
from itertools import chain
from django import forms
from django.contrib.postgres.validators import (
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from ..utils import prefix_validation_error
def _remove_trailing_nulls(self, values):
    index = None
    if self.remove_trailing_nulls:
        for i, value in reversed(list(enumerate(values))):
            if value in self.base_field.empty_values:
                index = i
            else:
                break
        if index is not None:
            values = values[:index]
    return (values, index)