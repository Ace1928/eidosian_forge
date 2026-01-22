import warnings
from django.core import exceptions
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin
class ForeignObjectRel(FieldCacheMixin):
    """
    Used by ForeignObject to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """
    auto_created = True
    concrete = False
    editable = False
    is_relation = True
    null = True
    empty_strings_allowed = False

    def __init__(self, field, to, related_name=None, related_query_name=None, limit_choices_to=None, parent_link=False, on_delete=None):
        self.field = field
        self.model = to
        self.related_name = related_name
        self.related_query_name = related_query_name
        self.limit_choices_to = {} if limit_choices_to is None else limit_choices_to
        self.parent_link = parent_link
        self.on_delete = on_delete
        self.symmetrical = False
        self.multiple = True

    @cached_property
    def hidden(self):
        return self.is_hidden()

    @cached_property
    def name(self):
        return self.field.related_query_name()

    @property
    def remote_field(self):
        return self.field

    @property
    def target_field(self):
        """
        When filtering against this relation, return the field on the remote
        model against which the filtering should happen.
        """
        target_fields = self.path_infos[-1].target_fields
        if len(target_fields) > 1:
            raise exceptions.FieldError("Can't use target_field for multicolumn relations.")
        return target_fields[0]

    @cached_property
    def related_model(self):
        if not self.field.model:
            raise AttributeError("This property can't be accessed before self.field.contribute_to_class has been called.")
        return self.field.model

    @cached_property
    def many_to_many(self):
        return self.field.many_to_many

    @cached_property
    def many_to_one(self):
        return self.field.one_to_many

    @cached_property
    def one_to_many(self):
        return self.field.many_to_one

    @cached_property
    def one_to_one(self):
        return self.field.one_to_one

    def get_lookup(self, lookup_name):
        return self.field.get_lookup(lookup_name)

    def get_lookups(self):
        return self.field.get_lookups()

    def get_transform(self, name):
        return self.field.get_transform(name)

    def get_internal_type(self):
        return self.field.get_internal_type()

    @property
    def db_type(self):
        return self.field.db_type

    def __repr__(self):
        return '<%s: %s.%s>' % (type(self).__name__, self.related_model._meta.app_label, self.related_model._meta.model_name)

    @property
    def identity(self):
        return (self.field, self.model, self.related_name, self.related_query_name, make_hashable(self.limit_choices_to), self.parent_link, self.on_delete, self.symmetrical, self.multiple)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self):
        return hash(self.identity)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('path_infos', None)
        return state

    def get_choices(self, include_blank=True, blank_choice=BLANK_CHOICE_DASH, limit_choices_to=None, ordering=()):
        """
        Return choices with a default blank choices included, for use
        as <select> choices for this field.

        Analog of django.db.models.fields.Field.get_choices(), provided
        initially for utilization by RelatedFieldListFilter.
        """
        limit_choices_to = limit_choices_to or self.limit_choices_to
        qs = self.related_model._default_manager.complex_filter(limit_choices_to)
        if ordering:
            qs = qs.order_by(*ordering)
        return (blank_choice if include_blank else []) + [(x.pk, str(x)) for x in qs]

    def is_hidden(self):
        """Should the related object be hidden?"""
        return bool(self.related_name) and self.related_name[-1] == '+'

    def get_joining_columns(self):
        warnings.warn('ForeignObjectRel.get_joining_columns() is deprecated. Use get_joining_fields() instead.', RemovedInDjango60Warning)
        return self.field.get_reverse_joining_columns()

    def get_joining_fields(self):
        return self.field.get_reverse_joining_fields()

    def get_extra_restriction(self, alias, related_alias):
        return self.field.get_extra_restriction(related_alias, alias)

    def set_field_name(self):
        """
        Set the related field's name, this is not available until later stages
        of app loading, so set_field_name is called from
        set_attributes_from_rel()
        """
        self.field_name = None

    def get_accessor_name(self, model=None):
        opts = model._meta if model else self.related_model._meta
        model = model or self.related_model
        if self.multiple:
            if self.symmetrical and model == self.model:
                return None
        if self.related_name:
            return self.related_name
        return opts.model_name + ('_set' if self.multiple else '')

    def get_path_info(self, filtered_relation=None):
        if filtered_relation:
            return self.field.get_reverse_path_info(filtered_relation)
        else:
            return self.field.reverse_path_infos

    @cached_property
    def path_infos(self):
        return self.get_path_info()

    def get_cache_name(self):
        """
        Return the name of the cache key to use for storing an instance of the
        forward model on the reverse model.
        """
        return self.get_accessor_name()