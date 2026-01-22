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
def _get_path_info_with_parent(self, filtered_relation):
    """
        Return the path that joins the current model through any parent models.
        The idea is that if you have a GFK defined on a parent model then we
        need to join the parent model first, then the child model.
        """
    path = []
    opts = self.remote_field.model._meta.concrete_model._meta
    parent_opts = opts.get_field(self.object_id_field_name).model._meta
    target = parent_opts.pk
    path.append(PathInfo(from_opts=self.model._meta, to_opts=parent_opts, target_fields=(target,), join_field=self.remote_field, m2m=True, direct=False, filtered_relation=filtered_relation))
    parent_field_chain = []
    while parent_opts != opts:
        field = opts.get_ancestor_link(parent_opts.model)
        parent_field_chain.append(field)
        opts = field.remote_field.model._meta
    parent_field_chain.reverse()
    for field in parent_field_chain:
        path.extend(field.remote_field.path_infos)
    return path