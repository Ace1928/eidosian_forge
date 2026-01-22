import collections
from itertools import chain
from django.apps import apps
from django.conf import settings
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.utils import NotRelationField, flatten, get_fields_from_path
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import Combinable
from django.forms.models import BaseModelForm, BaseModelFormSet, _get_foreign_key
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils.module_loading import import_string
def _check_radio_fields(self, obj):
    """Check that `radio_fields` is a dictionary."""
    if not isinstance(obj.radio_fields, dict):
        return must_be('a dictionary', option='radio_fields', obj=obj, id='admin.E021')
    else:
        return list(chain.from_iterable((self._check_radio_fields_key(obj, field_name, 'radio_fields') + self._check_radio_fields_value(obj, val, 'radio_fields["%s"]' % field_name) for field_name, val in obj.radio_fields.items())))