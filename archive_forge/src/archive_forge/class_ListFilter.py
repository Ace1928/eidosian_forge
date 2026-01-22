import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class ListFilter:
    title = None
    template = 'admin/filter.html'

    def __init__(self, request, params, model, model_admin):
        self.request = request
        self.used_parameters = {}
        if self.title is None:
            raise ImproperlyConfigured("The list filter '%s' does not specify a 'title'." % self.__class__.__name__)

    def has_output(self):
        """
        Return True if some choices would be output for this filter.
        """
        raise NotImplementedError('subclasses of ListFilter must provide a has_output() method')

    def choices(self, changelist):
        """
        Return choices ready to be output in the template.

        `changelist` is the ChangeList to be displayed.
        """
        raise NotImplementedError('subclasses of ListFilter must provide a choices() method')

    def queryset(self, request, queryset):
        """
        Return the filtered queryset.
        """
        raise NotImplementedError('subclasses of ListFilter must provide a queryset() method')

    def expected_parameters(self):
        """
        Return the list of parameter names that are expected from the
        request's query string and that will be used by this filter.
        """
        raise NotImplementedError('subclasses of ListFilter must provide an expected_parameters() method')