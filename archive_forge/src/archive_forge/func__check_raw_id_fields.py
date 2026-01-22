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
def _check_raw_id_fields(self, obj):
    """Check that `raw_id_fields` only contains field names that are listed
        on the model."""
    if not isinstance(obj.raw_id_fields, (list, tuple)):
        return must_be('a list or tuple', option='raw_id_fields', obj=obj, id='admin.E001')
    else:
        return list(chain.from_iterable((self._check_raw_id_fields_item(obj, field_name, 'raw_id_fields[%d]' % index) for index, field_name in enumerate(obj.raw_id_fields))))