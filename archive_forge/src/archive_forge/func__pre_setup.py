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
def _pre_setup(self):
    """
        Perform pre-test setup:
        * If the class has an 'available_apps' attribute, restrict the app
          registry to these applications, then fire the post_migrate signal --
          it must run with the correct set of applications for the test case.
        * If the class has a 'fixtures' attribute, install those fixtures.
        """
    super()._pre_setup()
    if self.available_apps is not None:
        apps.set_available_apps(self.available_apps)
        setting_changed.send(sender=settings._wrapped.__class__, setting='INSTALLED_APPS', value=self.available_apps, enter=True)
        for db_name in self._databases_names(include_mirrors=False):
            emit_post_migrate_signal(verbosity=0, interactive=False, db=db_name)
    try:
        self._fixture_setup()
    except Exception:
        if self.available_apps is not None:
            apps.unset_available_apps()
            setting_changed.send(sender=settings._wrapped.__class__, setting='INSTALLED_APPS', value=settings.INSTALLED_APPS, enter=False)
        raise
    for db_name in self._databases_names(include_mirrors=False):
        connections[db_name].queries_log.clear()