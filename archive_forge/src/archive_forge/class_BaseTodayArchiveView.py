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
class BaseTodayArchiveView(BaseDayArchiveView):
    """List of objects published today."""

    def get_dated_items(self):
        """Return (date_list, items, extra_context) for this request."""
        return self._get_dated_items(datetime.date.today())