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
def assertJSONNotEqual(self, raw, expected_data, msg=None):
    """
        Assert that the JSON fragments raw and expected_data are not equal.
        Usual JSON non-significant whitespace rules apply as the heavyweight
        is delegated to the json library.
        """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        self.fail('First argument is not valid JSON: %r' % raw)
    if isinstance(expected_data, str):
        try:
            expected_data = json.loads(expected_data)
        except json.JSONDecodeError:
            self.fail('Second argument is not valid JSON: %r' % expected_data)
    self.assertNotEqual(data, expected_data, msg=msg)