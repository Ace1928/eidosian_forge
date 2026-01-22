import copy
import datetime
from django.core.exceptions import NON_FIELD_ERRORS, ValidationError
from django.forms.fields import Field, FileField
from django.forms.utils import ErrorDict, ErrorList, RenderableFormMixin
from django.forms.widgets import Media, MediaDefiningClass
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.translation import gettext as _
from .renderers import get_default_renderer
class DeclarativeFieldsMetaclass(MediaDefiningClass):
    """Collect Fields declared on the base classes."""

    def __new__(mcs, name, bases, attrs):
        attrs['declared_fields'] = {key: attrs.pop(key) for key, value in list(attrs.items()) if isinstance(value, Field)}
        new_class = super().__new__(mcs, name, bases, attrs)
        declared_fields = {}
        for base in reversed(new_class.__mro__):
            if hasattr(base, 'declared_fields'):
                declared_fields.update(base.declared_fields)
            for attr, value in base.__dict__.items():
                if value is None and attr in declared_fields:
                    declared_fields.pop(attr)
        new_class.base_fields = declared_fields
        new_class.declared_fields = declared_fields
        return new_class