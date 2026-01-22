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
class DecimalField(Field):
    empty_strings_allowed = False
    default_error_messages = {'invalid': _('“%(value)s” value must be a decimal number.')}
    description = _('Decimal number')

    def __init__(self, verbose_name=None, name=None, max_digits=None, decimal_places=None, **kwargs):
        self.max_digits, self.decimal_places = (max_digits, decimal_places)
        super().__init__(verbose_name, name, **kwargs)

    def check(self, **kwargs):
        errors = super().check(**kwargs)
        digits_errors = [*self._check_decimal_places(), *self._check_max_digits()]
        if not digits_errors:
            errors.extend(self._check_decimal_places_and_max_digits(**kwargs))
        else:
            errors.extend(digits_errors)
        return errors

    def _check_decimal_places(self):
        try:
            decimal_places = int(self.decimal_places)
            if decimal_places < 0:
                raise ValueError()
        except TypeError:
            return [checks.Error("DecimalFields must define a 'decimal_places' attribute.", obj=self, id='fields.E130')]
        except ValueError:
            return [checks.Error("'decimal_places' must be a non-negative integer.", obj=self, id='fields.E131')]
        else:
            return []

    def _check_max_digits(self):
        try:
            max_digits = int(self.max_digits)
            if max_digits <= 0:
                raise ValueError()
        except TypeError:
            return [checks.Error("DecimalFields must define a 'max_digits' attribute.", obj=self, id='fields.E132')]
        except ValueError:
            return [checks.Error("'max_digits' must be a positive integer.", obj=self, id='fields.E133')]
        else:
            return []

    def _check_decimal_places_and_max_digits(self, **kwargs):
        if int(self.decimal_places) > int(self.max_digits):
            return [checks.Error("'max_digits' must be greater or equal to 'decimal_places'.", obj=self, id='fields.E134')]
        return []

    @cached_property
    def validators(self):
        return super().validators + [validators.DecimalValidator(self.max_digits, self.decimal_places)]

    @cached_property
    def context(self):
        return decimal.Context(prec=self.max_digits)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.max_digits is not None:
            kwargs['max_digits'] = self.max_digits
        if self.decimal_places is not None:
            kwargs['decimal_places'] = self.decimal_places
        return (name, path, args, kwargs)

    def get_internal_type(self):
        return 'DecimalField'

    def to_python(self, value):
        if value is None:
            return value
        try:
            if isinstance(value, float):
                decimal_value = self.context.create_decimal_from_float(value)
            else:
                decimal_value = decimal.Decimal(value)
        except (decimal.InvalidOperation, TypeError, ValueError):
            raise exceptions.ValidationError(self.error_messages['invalid'], code='invalid', params={'value': value})
        if not decimal_value.is_finite():
            raise exceptions.ValidationError(self.error_messages['invalid'], code='invalid', params={'value': value})
        return decimal_value

    def get_db_prep_save(self, value, connection):
        if hasattr(value, 'as_sql'):
            return value
        return connection.ops.adapt_decimalfield_value(self.to_python(value), self.max_digits, self.decimal_places)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        return self.to_python(value)

    def formfield(self, **kwargs):
        return super().formfield(**{'max_digits': self.max_digits, 'decimal_places': self.decimal_places, 'form_class': forms.DecimalField, **kwargs})