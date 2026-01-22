import functools
import itertools
import warnings
from collections import defaultdict
from asgiref.sync import sync_to_async
from django.contrib.contenttypes.models import ContentType
from django.core import checks
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import DEFAULT_DB_ALIAS, models, router, transaction
from django.db.models import DO_NOTHING, ForeignObject, ForeignObjectRel
from django.db.models.base import ModelBase, make_foreign_order_accessors
from django.db.models.fields.mixins import FieldCacheMixin
from django.db.models.fields.related import (
from django.db.models.query_utils import PathInfo
from django.db.models.sql import AND
from django.db.models.sql.where import WhereNode
from django.db.models.utils import AltersData
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
def _check_content_type_field(self):
    """
        Check if field named `field_name` in model `model` exists and is a
        valid content_type field (is a ForeignKey to ContentType).
        """
    try:
        field = self.model._meta.get_field(self.ct_field)
    except FieldDoesNotExist:
        return [checks.Error("The GenericForeignKey content type references the nonexistent field '%s.%s'." % (self.model._meta.object_name, self.ct_field), obj=self, id='contenttypes.E002')]
    else:
        if not isinstance(field, models.ForeignKey):
            return [checks.Error("'%s.%s' is not a ForeignKey." % (self.model._meta.object_name, self.ct_field), hint="GenericForeignKeys must use a ForeignKey to 'contenttypes.ContentType' as the 'content_type' field.", obj=self, id='contenttypes.E003')]
        elif field.remote_field.model != ContentType:
            return [checks.Error("'%s.%s' is not a ForeignKey to 'contenttypes.ContentType'." % (self.model._meta.object_name, self.ct_field), hint="GenericForeignKeys must use a ForeignKey to 'contenttypes.ContentType' as the 'content_type' field.", obj=self, id='contenttypes.E004')]
        else:
            return []