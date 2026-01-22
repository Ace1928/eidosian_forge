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
class YearMixin:
    """Mixin for views manipulating year-based data."""
    year_format = '%Y'
    year = None

    def get_year_format(self):
        """
        Get a year format string in strptime syntax to be used to parse the
        year from url variables.
        """
        return self.year_format

    def get_year(self):
        """Return the year for which this view should display data."""
        year = self.year
        if year is None:
            try:
                year = self.kwargs['year']
            except KeyError:
                try:
                    year = self.request.GET['year']
                except KeyError:
                    raise Http404(_('No year specified'))
        return year

    def get_next_year(self, date):
        """Get the next valid year."""
        return _get_next_prev(self, date, is_previous=False, period='year')

    def get_previous_year(self, date):
        """Get the previous valid year."""
        return _get_next_prev(self, date, is_previous=True, period='year')

    def _get_next_year(self, date):
        """
        Return the start date of the next interval.

        The interval is defined by start date <= item date < next start date.
        """
        try:
            return date.replace(year=date.year + 1, month=1, day=1)
        except ValueError:
            raise Http404(_('Date out of range'))

    def _get_current_year(self, date):
        """Return the start date of the current interval."""
        return date.replace(month=1, day=1)