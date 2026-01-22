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
class BaseDateDetailView(YearMixin, MonthMixin, DayMixin, DateMixin, BaseDetailView):
    """
    Detail view of a single object on a single date; this differs from the
    standard DetailView by accepting a year/month/day in the URL.
    """

    def get_object(self, queryset=None):
        """Get the object this request displays."""
        year = self.get_year()
        month = self.get_month()
        day = self.get_day()
        date = _date_from_string(year, self.get_year_format(), month, self.get_month_format(), day, self.get_day_format())
        qs = self.get_queryset() if queryset is None else queryset
        if not self.get_allow_future() and date > datetime.date.today():
            raise Http404(_('Future %(verbose_name_plural)s not available because %(class_name)s.allow_future is False.') % {'verbose_name_plural': qs.model._meta.verbose_name_plural, 'class_name': self.__class__.__name__})
        lookup_kwargs = self._make_single_date_lookup(date)
        qs = qs.filter(**lookup_kwargs)
        return super().get_object(queryset=qs)