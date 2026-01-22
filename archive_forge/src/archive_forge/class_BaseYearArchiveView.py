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
class BaseYearArchiveView(YearMixin, BaseDateListView):
    """List of objects published in a given year."""
    date_list_period = 'month'
    make_object_list = False

    def get_dated_items(self):
        """Return (date_list, items, extra_context) for this request."""
        year = self.get_year()
        date_field = self.get_date_field()
        date = _date_from_string(year, self.get_year_format())
        since = self._make_date_lookup_arg(date)
        until = self._make_date_lookup_arg(self._get_next_year(date))
        lookup_kwargs = {'%s__gte' % date_field: since, '%s__lt' % date_field: until}
        qs = self.get_dated_queryset(**lookup_kwargs)
        date_list = self.get_date_list(qs)
        if not self.get_make_object_list():
            qs = qs.none()
        return (date_list, qs, {'year': date, 'next_year': self.get_next_year(date), 'previous_year': self.get_previous_year(date)})

    def get_make_object_list(self):
        """
        Return `True` if this view should contain the full list of objects in
        the given year.
        """
        return self.make_object_list