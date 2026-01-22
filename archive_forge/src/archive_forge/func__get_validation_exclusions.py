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