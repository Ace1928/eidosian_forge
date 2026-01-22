import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class SimpleListFilter(FacetsMixin, ListFilter):
    parameter_name = None

    def __init__(self, request, params, model, model_admin):
        super().__init__(request, params, model, model_admin)
        if self.parameter_name is None:
            raise ImproperlyConfigured("The list filter '%s' does not specify a 'parameter_name'." % self.__class__.__name__)
        if self.parameter_name in params:
            value = params.pop(self.parameter_name)
            self.used_parameters[self.parameter_name] = value[-1]
        lookup_choices = self.lookups(request, model_admin)
        if lookup_choices is None:
            lookup_choices = ()
        self.lookup_choices = list(lookup_choices)

    def has_output(self):
        return len(self.lookup_choices) > 0

    def value(self):
        """
        Return the value (in string format) provided in the request's
        query string for this filter, if any, or None if the value wasn't
        provided.
        """
        return self.used_parameters.get(self.parameter_name)

    def lookups(self, request, model_admin):
        """
        Must be overridden to return a list of tuples (value, verbose value)
        """
        raise NotImplementedError('The SimpleListFilter.lookups() method must be overridden to return a list of tuples (value, verbose value).')

    def expected_parameters(self):
        return [self.parameter_name]

    def get_facet_counts(self, pk_attname, filtered_qs):
        original_value = self.used_parameters.get(self.parameter_name)
        counts = {}
        for i, choice in enumerate(self.lookup_choices):
            self.used_parameters[self.parameter_name] = choice[0]
            lookup_qs = self.queryset(self.request, filtered_qs)
            if lookup_qs is not None:
                counts[f'{i}__c'] = models.Count(pk_attname, filter=models.Q(pk__in=lookup_qs))
        self.used_parameters[self.parameter_name] = original_value
        return counts

    def choices(self, changelist):
        add_facets = changelist.add_facets
        facet_counts = self.get_facet_queryset(changelist) if add_facets else None
        yield {'selected': self.value() is None, 'query_string': changelist.get_query_string(remove=[self.parameter_name]), 'display': _('All')}
        for i, (lookup, title) in enumerate(self.lookup_choices):
            if add_facets:
                if (count := facet_counts.get(f'{i}__c', -1)) != -1:
                    title = f'{title} ({count})'
                else:
                    title = f'{title} (-)'
            yield {'selected': self.value() == str(lookup), 'query_string': changelist.get_query_string({self.parameter_name: lookup}), 'display': title}