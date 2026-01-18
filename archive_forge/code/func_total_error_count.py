from django.core.exceptions import ValidationError
from django.forms import Form
from django.forms.fields import BooleanField, IntegerField
from django.forms.renderers import get_default_renderer
from django.forms.utils import ErrorList, RenderableFormMixin
from django.forms.widgets import CheckboxInput, HiddenInput, NumberInput
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
def total_error_count(self):
    """Return the number of errors across all forms in the formset."""
    return len(self.non_form_errors()) + sum((len(form_errors) for form_errors in self.errors))