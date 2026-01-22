import json
from collections import UserList
from django.conf import settings
from django.core.exceptions import ValidationError
from django.forms.renderers import get_default_renderer
from django.utils import timezone
from django.utils.html import escape, format_html_join
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
class RenderableMixin:

    def get_context(self):
        raise NotImplementedError('Subclasses of RenderableMixin must provide a get_context() method.')

    def render(self, template_name=None, context=None, renderer=None):
        renderer = renderer or self.renderer
        template = template_name or self.template_name
        context = context or self.get_context()
        return mark_safe(renderer.render(template, context))
    __str__ = render
    __html__ = render