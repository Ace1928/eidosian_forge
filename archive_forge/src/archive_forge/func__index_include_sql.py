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
def _index_include_sql(self, model, columns):
    if not columns or not self.connection.features.supports_covering_indexes:
        return ''
    return Statement(' INCLUDE (%(columns)s)', columns=Columns(model._meta.db_table, columns, self.quote_name))