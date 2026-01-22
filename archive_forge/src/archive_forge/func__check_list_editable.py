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
def _check_list_editable(self, obj):
    """Check that list_editable is a sequence of editable fields from
        list_display without first element."""
    if not isinstance(obj.list_editable, (list, tuple)):
        return must_be('a list or tuple', option='list_editable', obj=obj, id='admin.E120')
    else:
        return list(chain.from_iterable((self._check_list_editable_item(obj, item, 'list_editable[%d]' % index) for index, item in enumerate(obj.list_editable))))