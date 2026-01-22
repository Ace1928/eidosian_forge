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
def _create_primary_key_sql(self, model, field):
    return Statement(self.sql_create_pk, table=Table(model._meta.db_table, self.quote_name), name=self.quote_name(self._create_index_name(model._meta.db_table, [field.column], suffix='_pk')), columns=Columns(model._meta.db_table, [field.column], self.quote_name))