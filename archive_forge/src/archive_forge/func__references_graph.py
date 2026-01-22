import datetime
import decimal
import uuid
from functools import lru_cache
from itertools import chain
from django.conf import settings
from django.core.exceptions import FieldError
from django.db import DatabaseError, NotSupportedError, models
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.constants import OnConflict
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.functional import cached_property
@cached_property
def _references_graph(self):
    return lru_cache(maxsize=512)(self.__references_graph)