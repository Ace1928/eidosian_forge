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
def _check_list_display_links_item(self, obj, field_name, label):
    if field_name not in obj.list_display:
        return [checks.Error("The value of '%s' refers to '%s', which is not defined in 'list_display'." % (label, field_name), obj=obj.__class__, id='admin.E111')]
    else:
        return []