import copy
import datetime
import decimal
import operator
import uuid
import warnings
from base64 import b64decode, b64encode
from functools import partialmethod, total_ordering
from django import forms
from django.apps import apps
from django.conf import settings
from django.core import checks, exceptions, validators
from django.db import connection, connections, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import DeferredAttribute, RegisterLookupMixin
from django.utils import timezone
from django.utils.choices import (
from django.utils.datastructures import DictWrapper
from django.utils.dateparse import (
from django.utils.duration import duration_microseconds, duration_string
from django.utils.functional import Promise, cached_property
from django.utils.ipv6 import clean_ipv6_address
from django.utils.itercompat import is_iterable
from django.utils.text import capfirst
from django.utils.translation import gettext_lazy as _
class FilePathField(Field):
    description = _('File path')

    def __init__(self, verbose_name=None, name=None, path='', match=None, recursive=False, allow_files=True, allow_folders=False, **kwargs):
        self.path, self.match, self.recursive = (path, match, recursive)
        self.allow_files, self.allow_folders = (allow_files, allow_folders)
        kwargs.setdefault('max_length', 100)
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        return [*super().check(**kwargs), *self._check_allowing_files_or_folders(**kwargs)]

    def _check_allowing_files_or_folders(self, **kwargs):
        if not self.allow_files and (not self.allow_folders):
            return [checks.Error("FilePathFields must have either 'allow_files' or 'allow_folders' set to True.", obj=self, id='fields.E140')]
        return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.path != '':
            kwargs['path'] = self.path
        if self.match is not None:
            kwargs['match'] = self.match
        if self.recursive is not False:
            kwargs['recursive'] = self.recursive
        if self.allow_files is not True:
            kwargs['allow_files'] = self.allow_files
        if self.allow_folders is not False:
            kwargs['allow_folders'] = self.allow_folders
        if kwargs.get('max_length') == 100:
            del kwargs['max_length']
        return (name, path, args, kwargs)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return None
        return str(value)

    def formfield(self, **kwargs):
        return super().formfield(**{'path': self.path() if callable(self.path) else self.path, 'match': self.match, 'recursive': self.recursive, 'form_class': forms.FilePathField, 'allow_files': self.allow_files, 'allow_folders': self.allow_folders, **kwargs})

    def get_internal_type(self):
        return 'FilePathField'