from pathlib import Path
import jinja2
from django.conf import settings
from django.template import TemplateDoesNotExist, TemplateSyntaxError
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from .base import BaseEngine
from .utils import csrf_input_lazy, csrf_token_lazy

    Format exception information for display on the debug page using the
    structure described in the template API documentation.
    