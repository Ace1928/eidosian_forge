import re
from django.core.exceptions import ValidationError
from django.forms.utils import RenderableFieldMixin, pretty_name
from django.forms.widgets import MultiWidget, Textarea, TextInput
from django.utils.functional import cached_property
from django.utils.html import format_html, html_safe
from django.utils.translation import gettext_lazy as _
@property
def use_fieldset(self):
    """
        Return the value of this BoundField widget's use_fieldset attribute.
        """
    return self.field.widget.use_fieldset