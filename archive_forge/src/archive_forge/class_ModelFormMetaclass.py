from itertools import chain
from django.core.exceptions import (
from django.db.models.utils import AltersData
from django.forms.fields import ChoiceField, Field
from django.forms.forms import BaseForm, DeclarativeFieldsMetaclass
from django.forms.formsets import BaseFormSet, formset_factory
from django.forms.utils import ErrorList
from django.forms.widgets import (
from django.utils.choices import BaseChoiceIterator
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
class ModelFormMetaclass(DeclarativeFieldsMetaclass):

    def __new__(mcs, name, bases, attrs):
        new_class = super().__new__(mcs, name, bases, attrs)
        if bases == (BaseModelForm,):
            return new_class
        opts = new_class._meta = ModelFormOptions(getattr(new_class, 'Meta', None))
        for opt in ['fields', 'exclude', 'localized_fields']:
            value = getattr(opts, opt)
            if isinstance(value, str) and value != ALL_FIELDS:
                msg = "%(model)s.Meta.%(opt)s cannot be a string. Did you mean to type: ('%(value)s',)?" % {'model': new_class.__name__, 'opt': opt, 'value': value}
                raise TypeError(msg)
        if opts.model:
            if opts.fields is None and opts.exclude is None:
                raise ImproperlyConfigured("Creating a ModelForm without either the 'fields' attribute or the 'exclude' attribute is prohibited; form %s needs updating." % name)
            if opts.fields == ALL_FIELDS:
                opts.fields = None
            fields = fields_for_model(opts.model, opts.fields, opts.exclude, opts.widgets, opts.formfield_callback, opts.localized_fields, opts.labels, opts.help_texts, opts.error_messages, opts.field_classes, apply_limit_choices_to=False, form_declared_fields=new_class.declared_fields)
            none_model_fields = {k for k, v in fields.items() if not v}
            missing_fields = none_model_fields.difference(new_class.declared_fields)
            if missing_fields:
                message = 'Unknown field(s) (%s) specified for %s'
                message %= (', '.join(missing_fields), opts.model.__name__)
                raise FieldError(message)
            fields.update(new_class.declared_fields)
        else:
            fields = new_class.declared_fields
        new_class.base_fields = fields
        return new_class