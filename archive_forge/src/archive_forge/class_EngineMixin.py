import functools
import warnings
from pathlib import Path
from django.conf import settings
from django.template.backends.django import DjangoTemplates
from django.template.loader import get_template
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class EngineMixin:

    def get_template(self, template_name):
        return self.engine.get_template(template_name)

    @cached_property
    def engine(self):
        return self.backend({'APP_DIRS': True, 'DIRS': [Path(__file__).parent / self.backend.app_dirname], 'NAME': 'djangoforms', 'OPTIONS': {}})