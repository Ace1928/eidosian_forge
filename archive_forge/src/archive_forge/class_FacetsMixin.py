import datetime
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.options import IncorrectLookupParameters
from django.contrib.admin.utils import (
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
class FacetsMixin:

    def get_facet_counts(self, pk_attname, filtered_qs):
        raise NotImplementedError('subclasses of FacetsMixin must provide a get_facet_counts() method.')

    def get_facet_queryset(self, changelist):
        filtered_qs = changelist.get_queryset(self.request, exclude_parameters=self.expected_parameters())
        return filtered_qs.aggregate(**self.get_facet_counts(changelist.pk_attname, filtered_qs))