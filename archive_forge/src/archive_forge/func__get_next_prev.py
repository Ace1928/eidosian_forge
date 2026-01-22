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
def _get_next_prev(generic_view, date, is_previous, period):
    """
    Get the next or the previous valid date. The idea is to allow links on
    month/day views to never be 404s by never providing a date that'll be
    invalid for the given view.

    This is a bit complicated since it handles different intervals of time,
    hence the coupling to generic_view.

    However in essence the logic comes down to:

        * If allow_empty and allow_future are both true, this is easy: just
          return the naive result (just the next/previous day/week/month,
          regardless of object existence.)

        * If allow_empty is true, allow_future is false, and the naive result
          isn't in the future, then return it; otherwise return None.

        * If allow_empty is false and allow_future is true, return the next
          date *that contains a valid object*, even if it's in the future. If
          there are no next objects, return None.

        * If allow_empty is false and allow_future is false, return the next
          date that contains a valid object. If that date is in the future, or
          if there are no next objects, return None.
    """
    date_field = generic_view.get_date_field()
    allow_empty = generic_view.get_allow_empty()
    allow_future = generic_view.get_allow_future()
    get_current = getattr(generic_view, '_get_current_%s' % period)
    get_next = getattr(generic_view, '_get_next_%s' % period)
    start, end = (get_current(date), get_next(date))
    if allow_empty:
        if is_previous:
            result = get_current(start - datetime.timedelta(days=1))
        else:
            result = end
        if allow_future or result <= timezone_today():
            return result
        else:
            return None
    else:
        if is_previous:
            lookup = {'%s__lt' % date_field: generic_view._make_date_lookup_arg(start)}
            ordering = '-%s' % date_field
        else:
            lookup = {'%s__gte' % date_field: generic_view._make_date_lookup_arg(end)}
            ordering = date_field
        if not allow_future:
            if generic_view.uses_datetime_field:
                now = timezone.now()
            else:
                now = timezone_today()
            lookup['%s__lte' % date_field] = now
        qs = generic_view.get_queryset().filter(**lookup).order_by(ordering)
        try:
            result = getattr(qs[0], date_field)
        except IndexError:
            return None
        if generic_view.uses_datetime_field:
            if settings.USE_TZ:
                result = timezone.localtime(result)
            result = result.date()
        return get_current(result)