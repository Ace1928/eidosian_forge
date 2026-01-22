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
def _check_filter_vertical(self, obj):
    """Check that filter_vertical is a sequence of field names."""
    if not isinstance(obj.filter_vertical, (list, tuple)):
        return must_be('a list or tuple', option='filter_vertical', obj=obj, id='admin.E017')
    else:
        return list(chain.from_iterable((self._check_filter_item(obj, field_name, 'filter_vertical[%d]' % index) for index, field_name in enumerate(obj.filter_vertical))))