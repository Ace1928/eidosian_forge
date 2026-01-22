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
class BaseDayArchiveView(YearMixin, MonthMixin, DayMixin, BaseDateListView):
    """List of objects published on a given day."""

    def get_dated_items(self):
        """Return (date_list, items, extra_context) for this request."""
        year = self.get_year()
        month = self.get_month()
        day = self.get_day()
        date = _date_from_string(year, self.get_year_format(), month, self.get_month_format(), day, self.get_day_format())
        return self._get_dated_items(date)

    def _get_dated_items(self, date):
        """
        Do the actual heavy lifting of getting the dated items; this accepts a
        date object so that TodayArchiveView can be trivial.
        """
        lookup_kwargs = self._make_single_date_lookup(date)
        qs = self.get_dated_queryset(**lookup_kwargs)
        return (None, qs, {'day': date, 'previous_day': self.get_previous_day(date), 'next_day': self.get_next_day(date), 'previous_month': self.get_previous_month(date), 'next_month': self.get_next_month(date)})