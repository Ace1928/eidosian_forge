import functools
import inspect
import warnings
from functools import partial
from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
from .related_lookups import (
from .reverse_related import ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel
def _check_referencing_to_swapped_model(self):
    if self.remote_field.model not in self.opts.apps.get_models() and (not isinstance(self.remote_field.model, str)) and self.remote_field.model._meta.swapped:
        return [checks.Error("Field defines a relation with the model '%s', which has been swapped out." % self.remote_field.model._meta.label, hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable, obj=self, id='fields.E301')]
    return []