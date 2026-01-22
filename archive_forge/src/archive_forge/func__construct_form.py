from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.renderers import get_default_renderer
from django.forms.utils import ErrorList, RenderableFormMixin
from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
def _construct_form(self, i, **kwargs):
    """Instantiate and return the i-th form instance in a formset."""
    defaults = {'auto_id': self.auto_id, 'prefix': self.add_prefix(i), 'error_class': self.error_class, 'use_required_attribute': False, 'renderer': self.form_renderer}
    if self.is_bound:
        defaults['data'] = self.data
        defaults['files'] = self.files
    if self.initial and 'initial' not in kwargs:
        try:
            defaults['initial'] = self.initial[i]
        except IndexError:
            pass
    if i >= self.initial_form_count() and i >= self.min_num:
        defaults['empty_permitted'] = True
    defaults.update(kwargs)
    form = self.form(**defaults)
    self.add_fields(form, i)
    return form