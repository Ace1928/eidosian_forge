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
@classmethod
def _check_swappable(cls):
    """Check if the swapped model exists."""
    errors = []
    if cls._meta.swapped:
        try:
            apps.get_model(cls._meta.swapped)
        except ValueError:
            errors.append(checks.Error("'%s' is not of the form 'app_label.app_name'." % cls._meta.swappable, id='models.E001'))
        except LookupError:
            app_label, model_name = cls._meta.swapped.split('.')
            errors.append(checks.Error("'%s' references '%s.%s', which has not been installed, or is abstract." % (cls._meta.swappable, app_label, model_name), id='models.E002'))
    return errors