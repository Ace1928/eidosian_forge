import functools
import inspect
import warnings
from functools import partial
from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
from .related_lookups import (
from .reverse_related import ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel
class RelatedField(FieldCacheMixin, Field):
    """Base class that all relational fields inherit from."""
    one_to_many = False
    one_to_one = False
    many_to_many = False
    many_to_one = False

    def __init__(self, related_name=None, related_query_name=None, limit_choices_to=None, **kwargs):
        self._related_name = related_name
        self._related_query_name = related_query_name
        self._limit_choices_to = limit_choices_to
        super().__init__(**kwargs)

    @cached_property
    def related_model(self):
        apps.check_models_ready()
        return self.remote_field.model

    def check(self, **kwargs):
        return [*super().check(**kwargs), *self._check_related_name_is_valid(), *self._check_related_query_name_is_valid(), *self._check_relation_model_exists(), *self._check_referencing_to_swapped_model(), *self._check_clashes()]

    def _check_related_name_is_valid(self):
        import keyword
        related_name = self.remote_field.related_name
        if related_name is None:
            return []
        is_valid_id = not keyword.iskeyword(related_name) and related_name.isidentifier()
        if not (is_valid_id or related_name.endswith('+')):
            return [checks.Error("The name '%s' is invalid related_name for field %s.%s" % (self.remote_field.related_name, self.model._meta.object_name, self.name), hint="Related name must be a valid Python identifier or end with a '+'", obj=self, id='fields.E306')]
        return []

    def _check_related_query_name_is_valid(self):
        if self.remote_field.is_hidden():
            return []
        rel_query_name = self.related_query_name()
        errors = []
        if rel_query_name.endswith('_'):
            errors.append(checks.Error("Reverse query name '%s' must not end with an underscore." % rel_query_name, hint='Add or change a related_name or related_query_name argument for this field.', obj=self, id='fields.E308'))
        if LOOKUP_SEP in rel_query_name:
            errors.append(checks.Error("Reverse query name '%s' must not contain '%s'." % (rel_query_name, LOOKUP_SEP), hint='Add or change a related_name or related_query_name argument for this field.', obj=self, id='fields.E309'))
        return errors

    def _check_relation_model_exists(self):
        rel_is_missing = self.remote_field.model not in self.opts.apps.get_models()
        rel_is_string = isinstance(self.remote_field.model, str)
        model_name = self.remote_field.model if rel_is_string else self.remote_field.model._meta.object_name
        if rel_is_missing and (rel_is_string or not self.remote_field.model._meta.swapped):
            return [checks.Error("Field defines a relation with model '%s', which is either not installed, or is abstract." % model_name, obj=self, id='fields.E300')]
        return []

    def _check_referencing_to_swapped_model(self):
        if self.remote_field.model not in self.opts.apps.get_models() and (not isinstance(self.remote_field.model, str)) and self.remote_field.model._meta.swapped:
            return [checks.Error("Field defines a relation with the model '%s', which has been swapped out." % self.remote_field.model._meta.label, hint="Update the relation to point at 'settings.%s'." % self.remote_field.model._meta.swappable, obj=self, id='fields.E301')]
        return []

    def _check_clashes(self):
        """Check accessor and reverse query name clashes."""
        from django.db.models.base import ModelBase
        errors = []
        opts = self.model._meta
        if not isinstance(self.remote_field.model, ModelBase):
            return []
        rel_opts = self.remote_field.model._meta
        rel_is_hidden = self.remote_field.is_hidden()
        rel_name = self.remote_field.get_accessor_name()
        rel_query_name = self.related_query_name()
        field_name = '%s.%s' % (opts.label, self.name)
        potential_clashes = rel_opts.fields + rel_opts.many_to_many
        for clash_field in potential_clashes:
            clash_name = '%s.%s' % (rel_opts.label, clash_field.name)
            if not rel_is_hidden and clash_field.name == rel_name:
                errors.append(checks.Error(f"Reverse accessor '{rel_opts.object_name}.{rel_name}' for '{field_name}' clashes with field name '{clash_name}'.", hint="Rename field '%s', or add/change a related_name argument to the definition for field '%s'." % (clash_name, field_name), obj=self, id='fields.E302'))
            if clash_field.name == rel_query_name:
                errors.append(checks.Error("Reverse query name for '%s' clashes with field name '%s'." % (field_name, clash_name), hint="Rename field '%s', or add/change a related_name argument to the definition for field '%s'." % (clash_name, field_name), obj=self, id='fields.E303'))
        potential_clashes = (r for r in rel_opts.related_objects if r.field is not self)
        for clash_field in potential_clashes:
            clash_name = '%s.%s' % (clash_field.related_model._meta.label, clash_field.field.name)
            if not rel_is_hidden and clash_field.get_accessor_name() == rel_name:
                errors.append(checks.Error(f"Reverse accessor '{rel_opts.object_name}.{rel_name}' for '{field_name}' clashes with reverse accessor for '{clash_name}'.", hint="Add or change a related_name argument to the definition for '%s' or '%s'." % (field_name, clash_name), obj=self, id='fields.E304'))
            if clash_field.get_accessor_name() == rel_query_name:
                errors.append(checks.Error("Reverse query name for '%s' clashes with reverse query name for '%s'." % (field_name, clash_name), hint="Add or change a related_name argument to the definition for '%s' or '%s'." % (field_name, clash_name), obj=self, id='fields.E305'))
        return errors

    def db_type(self, connection):
        return None

    def contribute_to_class(self, cls, name, private_only=False, **kwargs):
        super().contribute_to_class(cls, name, private_only=private_only, **kwargs)
        self.opts = cls._meta
        if not cls._meta.abstract:
            if self.remote_field.related_name:
                related_name = self.remote_field.related_name
            else:
                related_name = self.opts.default_related_name
            if related_name:
                related_name %= {'class': cls.__name__.lower(), 'model_name': cls._meta.model_name.lower(), 'app_label': cls._meta.app_label.lower()}
                self.remote_field.related_name = related_name
            if self.remote_field.related_query_name:
                related_query_name = self.remote_field.related_query_name % {'class': cls.__name__.lower(), 'app_label': cls._meta.app_label.lower()}
                self.remote_field.related_query_name = related_query_name

            def resolve_related_class(model, related, field):
                field.remote_field.model = related
                field.do_related_class(related, model)
            lazy_related_operation(resolve_related_class, cls, self.remote_field.model, field=self)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self._limit_choices_to:
            kwargs['limit_choices_to'] = self._limit_choices_to
        if self._related_name is not None:
            kwargs['related_name'] = self._related_name
        if self._related_query_name is not None:
            kwargs['related_query_name'] = self._related_query_name
        return (name, path, args, kwargs)

    def get_forward_related_filter(self, obj):
        """
        Return the keyword arguments that when supplied to
        self.model.object.filter(), would select all instances related through
        this field to the remote obj. This is used to build the querysets
        returned by related descriptors. obj is an instance of
        self.related_field.model.
        """
        return {'%s__%s' % (self.name, rh_field.name): getattr(obj, rh_field.attname) for _, rh_field in self.related_fields}

    def get_reverse_related_filter(self, obj):
        """
        Complement to get_forward_related_filter(). Return the keyword
        arguments that when passed to self.related_field.model.object.filter()
        select all instances of self.related_field.model related through
        this field to obj. obj is an instance of self.model.
        """
        base_q = Q.create([(rh_field.attname, getattr(obj, lh_field.attname)) for lh_field, rh_field in self.related_fields])
        descriptor_filter = self.get_extra_descriptor_filter(obj)
        if isinstance(descriptor_filter, dict):
            return base_q & Q(**descriptor_filter)
        elif descriptor_filter:
            return base_q & descriptor_filter
        return base_q

    @property
    def swappable_setting(self):
        """
        Get the setting that this is powered from for swapping, or None
        if it's not swapped in / marked with swappable=False.
        """
        if self.swappable:
            if isinstance(self.remote_field.model, str):
                to_string = self.remote_field.model
            else:
                to_string = self.remote_field.model._meta.label
            return apps.get_swappable_settings_name(to_string)
        return None

    def set_attributes_from_rel(self):
        self.name = self.name or self.remote_field.model._meta.model_name + '_' + self.remote_field.model._meta.pk.name
        if self.verbose_name is None:
            self.verbose_name = self.remote_field.model._meta.verbose_name
        self.remote_field.set_field_name()

    def do_related_class(self, other, cls):
        self.set_attributes_from_rel()
        self.contribute_to_related_class(other, self.remote_field)

    def get_limit_choices_to(self):
        """
        Return ``limit_choices_to`` for this model field.

        If it is a callable, it will be invoked and the result will be
        returned.
        """
        if callable(self.remote_field.limit_choices_to):
            return self.remote_field.limit_choices_to()
        return self.remote_field.limit_choices_to

    def formfield(self, **kwargs):
        """
        Pass ``limit_choices_to`` to the field being constructed.

        Only passes it if there is a type that supports related fields.
        This is a similar strategy used to pass the ``queryset`` to the field
        being constructed.
        """
        defaults = {}
        if hasattr(self.remote_field, 'get_related_field'):
            limit_choices_to = self.remote_field.limit_choices_to
            defaults.update({'limit_choices_to': limit_choices_to})
        defaults.update(kwargs)
        return super().formfield(**defaults)

    def related_query_name(self):
        """
        Define the name that can be used to identify this related object in a
        table-spanning query.
        """
        return self.remote_field.related_query_name or self.remote_field.related_name or self.opts.model_name

    @property
    def target_field(self):
        """
        When filtering against this relation, return the field on the remote
        model against which the filtering should happen.
        """
        target_fields = self.path_infos[-1].target_fields
        if len(target_fields) > 1:
            raise exceptions.FieldError('The relation has multiple target fields, but only single target field was asked for')
        return target_fields[0]

    def get_cache_name(self):
        return self.name