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
def _column_generated_sql(self, field):
    """Return the SQL to use in a GENERATED ALWAYS clause."""
    expression_sql, params = field.generated_sql(self.connection)
    persistency_sql = 'STORED' if field.db_persist else 'VIRTUAL'
    if self.connection.features.requires_literal_defaults:
        expression_sql = expression_sql % tuple((self.quote_value(p) for p in params))
        params = ()
    return (f'GENERATED ALWAYS AS ({expression_sql}) {persistency_sql}', params)