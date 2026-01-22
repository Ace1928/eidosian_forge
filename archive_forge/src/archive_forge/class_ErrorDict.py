import json
from collections import UserList
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms.renderers import get_default_renderer
from django.utils import timezone
from django.utils.html import escape, format_html_join
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
class ErrorDict(dict, RenderableErrorMixin):
    """
    A collection of errors that knows how to display itself in various formats.

    The dictionary keys are the field names, and the values are the errors.
    """
    template_name = 'django/forms/errors/dict/default.html'
    template_name_text = 'django/forms/errors/dict/text.txt'
    template_name_ul = 'django/forms/errors/dict/ul.html'

    def __init__(self, *args, renderer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = renderer or get_default_renderer()

    def as_data(self):
        return {f: e.as_data() for f, e in self.items()}

    def get_json_data(self, escape_html=False):
        return {f: e.get_json_data(escape_html) for f, e in self.items()}

    def get_context(self):
        return {'errors': self.items(), 'error_class': 'errorlist'}