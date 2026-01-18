import functools
import warnings
from pathlib import Path
from django.conf import settings
from django.template.backends.django import DjangoTemplates
from django.template.loader import get_template
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string

    Load templates using template.loader.get_template() which is configured
    based on settings.TEMPLATES.
    