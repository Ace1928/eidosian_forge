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
def _check_autocomplete_fields_item(self, obj, field_name, label):
    """
        Check that an item in `autocomplete_fields` is a ForeignKey or a
        ManyToManyField and that the item has a related ModelAdmin with
        search_fields defined.
        """
    try:
        field = obj.model._meta.get_field(field_name)
    except FieldDoesNotExist:
        return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E037')
    else:
        if not field.many_to_many and (not isinstance(field, models.ForeignKey)):
            return must_be('a foreign key or a many-to-many field', option=label, obj=obj, id='admin.E038')
        try:
            related_admin = obj.admin_site.get_model_admin(field.remote_field.model)
        except NotRegistered:
            return [checks.Error('An admin for model "%s" has to be registered to be referenced by %s.autocomplete_fields.' % (field.remote_field.model.__name__, type(obj).__name__), obj=obj.__class__, id='admin.E039')]
        else:
            if not related_admin.search_fields:
                return [checks.Error('%s must define "search_fields", because it\'s referenced by %s.autocomplete_fields.' % (related_admin.__class__.__name__, type(obj).__name__), obj=obj.__class__, id='admin.E040')]
        return []