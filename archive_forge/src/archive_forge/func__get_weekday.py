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
def _get_weekday(self, date):
    """
        Return the weekday for a given date.

        The first day according to the week format is 0 and the last day is 6.
        """
    week_format = self.get_week_format()
    if week_format in {'%W', '%V'}:
        return date.weekday()
    elif week_format == '%U':
        return (date.weekday() + 1) % 7
    else:
        raise ValueError('unknown week format: %s' % week_format)