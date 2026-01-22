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
def _check_fieldsets_item(self, obj, fieldset, label, seen_fields):
    """Check an item of `fieldsets`, i.e. check that this is a pair of a
        set name and a dictionary containing "fields" key."""
    if not isinstance(fieldset, (list, tuple)):
        return must_be('a list or tuple', option=label, obj=obj, id='admin.E008')
    elif len(fieldset) != 2:
        return must_be('of length 2', option=label, obj=obj, id='admin.E009')
    elif not isinstance(fieldset[1], dict):
        return must_be('a dictionary', option='%s[1]' % label, obj=obj, id='admin.E010')
    elif 'fields' not in fieldset[1]:
        return [checks.Error("The value of '%s[1]' must contain the key 'fields'." % label, obj=obj.__class__, id='admin.E011')]
    elif not isinstance(fieldset[1]['fields'], (list, tuple)):
        return must_be('a list or tuple', option="%s[1]['fields']" % label, obj=obj, id='admin.E008')
    seen_fields.extend(flatten(fieldset[1]['fields']))
    if len(seen_fields) != len(set(seen_fields)):
        return [checks.Error("There are duplicate field(s) in '%s[1]'." % label, obj=obj.__class__, id='admin.E012')]
    return list(chain.from_iterable((self._check_field_spec(obj, fieldset_fields, '%s[1]["fields"]' % label) for fieldset_fields in fieldset[1]['fields'])))