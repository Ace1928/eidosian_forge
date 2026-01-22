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
class DayMixin:
    """Mixin for views manipulating day-based data."""
    day_format = '%d'
    day = None

    def get_day_format(self):
        """
        Get a day format string in strptime syntax to be used to parse the day
        from url variables.
        """
        return self.day_format

    def get_day(self):
        """Return the day for which this view should display data."""
        day = self.day
        if day is None:
            try:
                day = self.kwargs['day']
            except KeyError:
                try:
                    day = self.request.GET['day']
                except KeyError:
                    raise Http404(_('No day specified'))
        return day

    def get_next_day(self, date):
        """Get the next valid day."""
        return _get_next_prev(self, date, is_previous=False, period='day')

    def get_previous_day(self, date):
        """Get the previous valid day."""
        return _get_next_prev(self, date, is_previous=True, period='day')

    def _get_next_day(self, date):
        """
        Return the start date of the next interval.

        The interval is defined by start date <= item date < next start date.
        """
        return date + datetime.timedelta(days=1)

    def _get_current_day(self, date):
        """Return the start date of the current interval."""
        return date