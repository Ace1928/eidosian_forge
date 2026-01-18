import datetime
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import models
from django.http import Http404
from django.utils import timezone
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from django.views.generic.base import View
from django.views.generic.detail import (
from django.views.generic.list import (
@cached_property
def uses_datetime_field(self):
    """
        Return `True` if the date field is a `DateTimeField` and `False`
        if it's a `DateField`.
        """
    model = self.get_queryset().model if self.model is None else self.model
    field = model._meta.get_field(self.get_date_field())
    return isinstance(field, models.DateTimeField)