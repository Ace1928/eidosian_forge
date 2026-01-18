import copy
from itertools import chain
from django import forms
from django.contrib.postgres.validators import (
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from ..utils import prefix_validation_error
def value_omitted_from_data(self, data, files, name):
    return all((self.widget.value_omitted_from_data(data, files, '%s_%s' % (name, index)) for index in range(self.size)))