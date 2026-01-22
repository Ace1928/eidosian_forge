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
def _rename_index_sql(self, model, old_name, new_name):
    return Statement(self.sql_rename_index, table=Table(model._meta.db_table, self.quote_name), old_name=self.quote_name(old_name), new_name=self.quote_name(new_name))