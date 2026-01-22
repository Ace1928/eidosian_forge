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
def _iter_column_sql(self, column_db_type, params, model, field, field_db_params, include_default):
    yield column_db_type
    if (collation := field_db_params.get('collation')):
        yield self._collate_sql(collation)
    if self.connection.features.supports_comments_inline and field.db_comment:
        yield self._comment_sql(field.db_comment)
    null = field.null
    if field.db_default is not NOT_PROVIDED:
        default_sql, default_params = self.db_default_sql(field)
        yield f'DEFAULT {default_sql}'
        params.extend(default_params)
        include_default = False
    include_default = include_default and (not self.skip_default(field)) and (not (null and self.skip_default_on_alter(field)))
    if include_default:
        default_value = self.effective_default(field)
        if default_value is not None:
            column_default = 'DEFAULT ' + self._column_default_sql(field)
            if self.connection.features.requires_literal_defaults:
                yield (column_default % self.prepare_default(default_value))
            else:
                yield column_default
                params.append(default_value)
    if field.empty_strings_allowed and (not field.primary_key) and self.connection.features.interprets_empty_strings_as_nulls:
        null = True
    if field.generated:
        generated_sql, generated_params = self._column_generated_sql(field)
        params.extend(generated_params)
        yield generated_sql
    elif not null:
        yield 'NOT NULL'
    elif not self.connection.features.implied_column_null:
        yield 'NULL'
    if field.primary_key:
        yield 'PRIMARY KEY'
    elif field.unique:
        yield 'UNIQUE'
    tablespace = field.db_tablespace or model._meta.db_tablespace
    if tablespace and self.connection.features.supports_tablespaces and field.unique:
        yield self.connection.ops.tablespace_sql(tablespace, inline=True)