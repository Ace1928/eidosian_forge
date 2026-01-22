import itertools
import math
import warnings
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.expressions import Case, Expression, Func, Value, When
from django.db.models.fields import (
from django.db.models.query_utils import RegisterLookupMixin
from django.utils.datastructures import OrderedSet
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
class UUIDTextMixin:
    """
    Strip hyphens from a value when filtering a UUIDField on backends without
    a native datatype for UUID.
    """

    def process_rhs(self, qn, connection):
        if not connection.features.has_native_uuid_field:
            from django.db.models.functions import Replace
            if self.rhs_is_direct_value():
                self.rhs = Value(self.rhs)
            self.rhs = Replace(self.rhs, Value('-'), Value(''), output_field=CharField())
        rhs, params = super().process_rhs(qn, connection)
        return (rhs, params)