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
def _check_generic_foreign_key_existence(self):
    target = self.remote_field.model
    if isinstance(target, ModelBase):
        fields = target._meta.private_fields
        if any((self._is_matching_generic_foreign_key(field) for field in fields)):
            return []
        else:
            return [checks.Error("The GenericRelation defines a relation with the model '%s', but that model does not have a GenericForeignKey." % target._meta.label, obj=self, id='contenttypes.E004')]
    else:
        return []