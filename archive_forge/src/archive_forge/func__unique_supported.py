import logging
import operator
from datetime import datetime
from django.conf import settings
from django.db.backends.ddl_references import (
from django.db.backends.utils import names_digest, split_identifier, truncate_name
from django.db.models import NOT_PROVIDED, Deferrable, Index
from django.db.models.sql import Query
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone
def _unique_supported(self, condition=None, deferrable=None, include=None, expressions=None, nulls_distinct=None):
    return (not condition or self.connection.features.supports_partial_indexes) and (not deferrable or self.connection.features.supports_deferrable_unique_constraints) and (not include or self.connection.features.supports_covering_indexes) and (not expressions or self.connection.features.supports_expression_indexes) and (nulls_distinct is None or self.connection.features.supports_nulls_distinct_unique_constraints)