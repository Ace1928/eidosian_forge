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
def validate_constraints(self, exclude=None):
    constraints = self.get_constraints()
    using = router.db_for_write(self.__class__, instance=self)
    errors = {}
    for model_class, model_constraints in constraints:
        for constraint in model_constraints:
            try:
                constraint.validate(model_class, self, exclude=exclude, using=using)
            except ValidationError as e:
                if getattr(e, 'code', None) == 'unique' and len(constraint.fields) == 1:
                    errors.setdefault(constraint.fields[0], []).append(e)
                else:
                    errors = e.update_error_dict(errors)
    if errors:
        raise ValidationError(errors)