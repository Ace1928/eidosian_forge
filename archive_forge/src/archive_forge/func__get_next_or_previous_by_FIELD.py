import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
from asgiref.sync import sync_to_async
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
from django.db import (
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import DatabaseDefault, RawSQL
from django.db.models.fields.related import (
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
def _get_next_or_previous_by_FIELD(self, field, is_next, **kwargs):
    if not self.pk:
        raise ValueError('get_next/get_previous cannot be used on unsaved objects.')
    op = 'gt' if is_next else 'lt'
    order = '' if is_next else '-'
    param = getattr(self, field.attname)
    q = Q.create([(field.name, param), (f'pk__{op}', self.pk)], connector=Q.AND)
    q = Q.create([q, (f'{field.name}__{op}', param)], connector=Q.OR)
    qs = self.__class__._default_manager.using(self._state.db).filter(**kwargs).filter(q).order_by('%s%s' % (order, field.name), '%spk' % order)
    try:
        return qs[0]
    except IndexError:
        raise self.DoesNotExist('%s matching query does not exist.' % self.__class__._meta.object_name)