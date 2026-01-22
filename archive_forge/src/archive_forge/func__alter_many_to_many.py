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
def _alter_many_to_many(self, model, old_field, new_field, strict):
    """Alter M2Ms to repoint their to= endpoints."""
    if old_field.remote_field.through._meta.db_table != new_field.remote_field.through._meta.db_table:
        self.alter_db_table(old_field.remote_field.through, old_field.remote_field.through._meta.db_table, new_field.remote_field.through._meta.db_table)
    self.alter_field(new_field.remote_field.through, old_field.remote_field.through._meta.get_field(old_field.m2m_reverse_field_name()), new_field.remote_field.through._meta.get_field(new_field.m2m_reverse_field_name()))
    self.alter_field(new_field.remote_field.through, old_field.remote_field.through._meta.get_field(old_field.m2m_field_name()), new_field.remote_field.through._meta.get_field(new_field.m2m_field_name()))