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
@staticmethod
def _effective_default(field):
    if field.has_default():
        default = field.get_default()
    elif field.generated:
        default = None
    elif not field.null and field.blank and field.empty_strings_allowed:
        if field.get_internal_type() == 'BinaryField':
            default = b''
        else:
            default = ''
    elif getattr(field, 'auto_now', False) or getattr(field, 'auto_now_add', False):
        internal_type = field.get_internal_type()
        if internal_type == 'DateTimeField':
            default = timezone.now()
        else:
            default = datetime.now()
            if internal_type == 'DateField':
                default = default.date()
            elif internal_type == 'TimeField':
                default = default.time()
    else:
        default = None
    return default