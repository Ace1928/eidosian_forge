import bisect
import copy
import inspect
import warnings
from collections import defaultdict
from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import connections
from django.db.models import AutoField, Manager, OrderWrt, UniqueConstraint
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override
@cached_property
def total_unique_constraints(self):
    """
        Return a list of total unique constraints. Useful for determining set
        of fields guaranteed to be unique for all rows.
        """
    return [constraint for constraint in self.constraints if isinstance(constraint, UniqueConstraint) and constraint.condition is None and (not constraint.contains_expressions)]