from functools import partial
from django.contrib.admin.checks import InlineModelAdminChecks
from django.contrib.admin.options import InlineModelAdmin, flatten_fieldsets
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.forms import (
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.forms import ALL_FIELDS
from django.forms.models import modelform_defines_fields
def _check_exclude_of_parent_model(self, obj, parent_model):
    return []