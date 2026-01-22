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
def _check_list_display_item(self, obj, item, label):
    if callable(item):
        return []
    elif hasattr(obj, item):
        return []
    try:
        field = obj.model._meta.get_field(item)
    except FieldDoesNotExist:
        try:
            field = getattr(obj.model, item)
        except AttributeError:
            return [checks.Error("The value of '%s' refers to '%s', which is not a callable, an attribute of '%s', or an attribute or method on '%s'." % (label, item, obj.__class__.__name__, obj.model._meta.label), obj=obj.__class__, id='admin.E108')]
    if getattr(field, 'is_relation', False) and (field.many_to_many or field.one_to_many) or (getattr(field, 'rel', None) and field.rel.field.many_to_one):
        return [checks.Error(f"The value of '{label}' must not be a many-to-many field or a reverse foreign key.", obj=obj.__class__, id='admin.E109')]
    return []