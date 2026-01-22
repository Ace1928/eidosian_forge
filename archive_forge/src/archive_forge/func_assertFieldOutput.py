import difflib
import json
import logging
import posixpath
import sys
import threading
import unittest
import warnings
from collections import Counter
from contextlib import contextmanager
from copy import copy, deepcopy
from difflib import get_close_matches
from functools import wraps
from unittest.suite import _DebugResult
from unittest.util import safe_repr
from urllib.parse import (
from urllib.request import url2pathname
from asgiref.sync import async_to_sync, iscoroutinefunction
from django.apps import apps
from django.conf import settings
from django.core import mail
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.core.files import locks
from django.core.handlers.wsgi import WSGIHandler, get_path_info
from django.core.management import call_command
from django.core.management.color import no_style
from django.core.management.sql import emit_post_migrate_signal
from django.core.servers.basehttp import ThreadedWSGIServer, WSGIRequestHandler
from django.core.signals import setting_changed
from django.db import DEFAULT_DB_ALIAS, connection, connections, transaction
from django.forms.fields import CharField
from django.http import QueryDict
from django.http.request import split_domain_port, validate_host
from django.test.client import AsyncClient, Client
from django.test.html import HTMLParseError, parse_html
from django.test.signals import template_rendered
from django.test.utils import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import classproperty
from django.views.static import serve
def assertFieldOutput(self, fieldclass, valid, invalid, field_args=None, field_kwargs=None, empty_value=''):
    """
        Assert that a form field behaves correctly with various inputs.

        Args:
            fieldclass: the class of the field to be tested.
            valid: a dictionary mapping valid inputs to their expected
                    cleaned values.
            invalid: a dictionary mapping invalid inputs to one or more
                    raised error messages.
            field_args: the args passed to instantiate the field
            field_kwargs: the kwargs passed to instantiate the field
            empty_value: the expected clean output for inputs in empty_values
        """
    if field_args is None:
        field_args = []
    if field_kwargs is None:
        field_kwargs = {}
    required = fieldclass(*field_args, **field_kwargs)
    optional = fieldclass(*field_args, **{**field_kwargs, 'required': False})
    for input, output in valid.items():
        self.assertEqual(required.clean(input), output)
        self.assertEqual(optional.clean(input), output)
    for input, errors in invalid.items():
        with self.assertRaises(ValidationError) as context_manager:
            required.clean(input)
        self.assertEqual(context_manager.exception.messages, errors)
        with self.assertRaises(ValidationError) as context_manager:
            optional.clean(input)
        self.assertEqual(context_manager.exception.messages, errors)
    error_required = [required.error_messages['required']]
    for e in required.empty_values:
        with self.assertRaises(ValidationError) as context_manager:
            required.clean(e)
        self.assertEqual(context_manager.exception.messages, error_required)
        self.assertEqual(optional.clean(e), empty_value)
    if issubclass(fieldclass, CharField):
        field_kwargs.update({'min_length': 2, 'max_length': 20})
        self.assertIsInstance(fieldclass(*field_args, **field_kwargs), fieldclass)