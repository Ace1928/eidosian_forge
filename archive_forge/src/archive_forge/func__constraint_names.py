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
def _constraint_names(self, model, column_names=None, unique=None, primary_key=None, index=None, foreign_key=None, check=None, type_=None, exclude=None):
    """Return all constraint names matching the columns and conditions."""
    if column_names is not None:
        column_names = [self.connection.introspection.identifier_converter(truncate_name(name, self.connection.ops.max_name_length())) if self.connection.features.truncates_names else self.connection.introspection.identifier_converter(name) for name in column_names]
    with self.connection.cursor() as cursor:
        constraints = self.connection.introspection.get_constraints(cursor, model._meta.db_table)
    result = []
    for name, infodict in constraints.items():
        if column_names is None or column_names == infodict['columns']:
            if unique is not None and infodict['unique'] != unique:
                continue
            if primary_key is not None and infodict['primary_key'] != primary_key:
                continue
            if index is not None and infodict['index'] != index:
                continue
            if check is not None and infodict['check'] != check:
                continue
            if foreign_key is not None and (not infodict['foreign_key']):
                continue
            if type_ is not None and infodict['type'] != type_:
                continue
            if not exclude or name not in exclude:
                result.append(name)
    return result