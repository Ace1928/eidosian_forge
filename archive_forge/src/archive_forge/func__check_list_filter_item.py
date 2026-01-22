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
def _check_list_filter_item(self, obj, item, label):
    """
        Check one item of `list_filter`, i.e. check if it is one of three options:
        1. 'field' -- a basic field filter, possibly w/ relationships (e.g.
           'field__rel')
        2. ('field', SomeFieldListFilter) - a field-based list filter class
        3. SomeListFilter - a non-field list filter class
        """
    from django.contrib.admin import FieldListFilter, ListFilter
    if callable(item) and (not isinstance(item, models.Field)):
        if not _issubclass(item, ListFilter):
            return must_inherit_from(parent='ListFilter', option=label, obj=obj, id='admin.E113')
        elif issubclass(item, FieldListFilter):
            return [checks.Error("The value of '%s' must not inherit from 'FieldListFilter'." % label, obj=obj.__class__, id='admin.E114')]
        else:
            return []
    elif isinstance(item, (tuple, list)):
        field, list_filter_class = item
        if not _issubclass(list_filter_class, FieldListFilter):
            return must_inherit_from(parent='FieldListFilter', option='%s[1]' % label, obj=obj, id='admin.E115')
        else:
            return []
    else:
        field = item
        try:
            get_fields_from_path(obj.model, field)
        except (NotRelationField, FieldDoesNotExist):
            return [checks.Error("The value of '%s' refers to '%s', which does not refer to a Field." % (label, field), obj=obj.__class__, id='admin.E116')]
        else:
            return []