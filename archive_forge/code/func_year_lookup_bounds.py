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
def year_lookup_bounds(self, connection, year):
    from django.db.models.functions import ExtractIsoYear
    iso_year = isinstance(self.lhs, ExtractIsoYear)
    output_field = self.lhs.lhs.output_field
    if isinstance(output_field, DateTimeField):
        bounds = connection.ops.year_lookup_bounds_for_datetime_field(year, iso_year=iso_year)
    else:
        bounds = connection.ops.year_lookup_bounds_for_date_field(year, iso_year=iso_year)
    return bounds