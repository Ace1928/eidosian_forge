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
def _check_actions_uniqueness(self, obj):
    """Check that every action has a unique __name__."""
    errors = []
    names = collections.Counter((name for _, name, _ in obj._get_base_actions()))
    for name, count in names.items():
        if count > 1:
            errors.append(checks.Error('__name__ attributes of actions defined in %s must be unique. Name %r is not unique.' % (obj.__class__.__name__, name), obj=obj.__class__, id='admin.E130'))
    return errors