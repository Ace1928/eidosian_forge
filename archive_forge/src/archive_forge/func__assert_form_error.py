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
def _assert_form_error(self, form, field, errors, msg_prefix, form_repr):
    if not form.is_bound:
        self.fail(f'{msg_prefix}The {form_repr} is not bound, it will never have any errors.')
    if field is not None and field not in form.fields:
        self.fail(f'{msg_prefix}The {form_repr} does not contain the field {field!r}.')
    if field is None:
        field_errors = form.non_field_errors()
        failure_message = f"The non-field errors of {form_repr} don't match."
    else:
        field_errors = form.errors.get(field, [])
        failure_message = f"The errors of field {field!r} on {form_repr} don't match."
    self.assertEqual(field_errors, errors, msg_prefix + failure_message)