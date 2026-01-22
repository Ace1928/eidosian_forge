import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import FieldError
from django.db import (
from django.db.models import Q, Window, signals
from django.db.models.functions import RowNumber
from django.db.models.lookups import GreaterThan, LessThanOrEqual
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData, resolve_callables
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
def _check_fk_val(self):
    for field in self.field.foreign_related_fields:
        if getattr(self.instance, field.attname) is None:
            raise ValueError(f'"{self.instance!r}" needs to have a value for field "{field.attname}" before this relationship can be used.')