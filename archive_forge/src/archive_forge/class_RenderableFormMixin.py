import json
from collections import UserList
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms.renderers import get_default_renderer
from django.utils import timezone
from django.utils.html import escape, format_html_join
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
class RenderableFormMixin(RenderableMixin):

    def as_p(self):
        """Render as <p> elements."""
        return self.render(self.template_name_p)

    def as_table(self):
        """Render as <tr> elements excluding the surrounding <table> tag."""
        return self.render(self.template_name_table)

    def as_ul(self):
        """Render as <li> elements excluding the surrounding <ul> tag."""
        return self.render(self.template_name_ul)

    def as_div(self):
        """Render as <div> elements."""
        return self.render(self.template_name_div)