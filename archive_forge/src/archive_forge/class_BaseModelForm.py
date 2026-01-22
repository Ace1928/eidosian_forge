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
class BaseModelForm(BaseForm, AltersData):

    def __init__(self, data=None, files=None, auto_id='id_%s', prefix=None, initial=None, error_class=ErrorList, label_suffix=None, empty_permitted=False, instance=None, use_required_attribute=None, renderer=None):
        opts = self._meta
        if opts.model is None:
            raise ValueError('ModelForm has no model class specified.')
        if instance is None:
            self.instance = opts.model()
            object_data = {}
        else:
            self.instance = instance
            object_data = model_to_dict(instance, opts.fields, opts.exclude)
        if initial is not None:
            object_data.update(initial)
        self._validate_unique = False
        super().__init__(data, files, auto_id, prefix, object_data, error_class, label_suffix, empty_permitted, use_required_attribute=use_required_attribute, renderer=renderer)
        for formfield in self.fields.values():
            apply_limit_choices_to_to_formfield(formfield)

    def _get_validation_exclusions(self):
        """
        For backwards-compatibility, exclude several types of fields from model
        validation. See tickets #12507, #12521, #12553.
        """
        exclude = set()
        for f in self.instance._meta.fields:
            field = f.name
            if field not in self.fields:
                exclude.add(f.name)
            elif self._meta.fields and field not in self._meta.fields:
                exclude.add(f.name)
            elif self._meta.exclude and field in self._meta.exclude:
                exclude.add(f.name)
            elif field in self._errors:
                exclude.add(f.name)
            else:
                form_field = self.fields[field]
                field_value = self.cleaned_data.get(field)
                if not f.blank and (not form_field.required) and (field_value in form_field.empty_values):
                    exclude.add(f.name)
        return exclude

    def clean(self):
        self._validate_unique = True
        return self.cleaned_data

    def _update_errors(self, errors):
        opts = self._meta
        if hasattr(errors, 'error_dict'):
            error_dict = errors.error_dict
        else:
            error_dict = {NON_FIELD_ERRORS: errors}
        for field, messages in error_dict.items():
            if field == NON_FIELD_ERRORS and opts.error_messages and (NON_FIELD_ERRORS in opts.error_messages):
                error_messages = opts.error_messages[NON_FIELD_ERRORS]
            elif field in self.fields:
                error_messages = self.fields[field].error_messages
            else:
                continue
            for message in messages:
                if isinstance(message, ValidationError) and message.code in error_messages:
                    message.message = error_messages[message.code]
        self.add_error(None, errors)

    def _post_clean(self):
        opts = self._meta
        exclude = self._get_validation_exclusions()
        for name, field in self.fields.items():
            if isinstance(field, InlineForeignKeyField):
                exclude.add(name)
        try:
            self.instance = construct_instance(self, self.instance, opts.fields, opts.exclude)
        except ValidationError as e:
            self._update_errors(e)
        try:
            self.instance.full_clean(exclude=exclude, validate_unique=False)
        except ValidationError as e:
            self._update_errors(e)
        if self._validate_unique:
            self.validate_unique()

    def validate_unique(self):
        """
        Call the instance's validate_unique() method and update the form's
        validation errors if any were raised.
        """
        exclude = self._get_validation_exclusions()
        try:
            self.instance.validate_unique(exclude=exclude)
        except ValidationError as e:
            self._update_errors(e)

    def _save_m2m(self):
        """
        Save the many-to-many fields and generic relations for this form.
        """
        cleaned_data = self.cleaned_data
        exclude = self._meta.exclude
        fields = self._meta.fields
        opts = self.instance._meta
        for f in chain(opts.many_to_many, opts.private_fields):
            if not hasattr(f, 'save_form_data'):
                continue
            if fields and f.name not in fields:
                continue
            if exclude and f.name in exclude:
                continue
            if f.name in cleaned_data:
                f.save_form_data(self.instance, cleaned_data[f.name])

    def save(self, commit=True):
        """
        Save this form's self.instance object if commit=True. Otherwise, add
        a save_m2m() method to the form which can be called after the instance
        is saved manually at a later time. Return the model instance.
        """
        if self.errors:
            raise ValueError("The %s could not be %s because the data didn't validate." % (self.instance._meta.object_name, 'created' if self.instance._state.adding else 'changed'))
        if commit:
            self.instance.save()
            self._save_m2m()
        else:
            self.save_m2m = self._save_m2m
        return self.instance
    save.alters_data = True