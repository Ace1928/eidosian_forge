import json
from collections import UserList
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms.renderers import get_default_renderer
from django.utils import timezone
from django.utils.html import escape, format_html_join
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
class RenderableFieldMixin(RenderableMixin):

    def as_field_group(self):
        return self.render()

    def as_hidden(self):
        raise NotImplementedError('Subclasses of RenderableFieldMixin must provide an as_hidden() method.')

    def as_widget(self):
        raise NotImplementedError('Subclasses of RenderableFieldMixin must provide an as_widget() method.')

    def __str__(self):
        """Render this field as an HTML widget."""
        if self.field.show_hidden_initial:
            return self.as_widget() + self.as_hidden(only_initial=True)
        return self.as_widget()
    __html__ = __str__