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
def assertRedirects(self, response, expected_url, status_code=302, target_status_code=200, msg_prefix='', fetch_redirect_response=True):
    """
        Assert that a response redirected to a specific URL and that the
        redirect URL can be loaded.

        Won't work for external links since it uses the test client to do a
        request (use fetch_redirect_response=False to check such links without
        fetching them).
        """
    if msg_prefix:
        msg_prefix += ': '
    if hasattr(response, 'redirect_chain'):
        self.assertTrue(response.redirect_chain, msg_prefix + "Response didn't redirect as expected: Response code was %d (expected %d)" % (response.status_code, status_code))
        self.assertEqual(response.redirect_chain[0][1], status_code, msg_prefix + "Initial response didn't redirect as expected: Response code was %d (expected %d)" % (response.redirect_chain[0][1], status_code))
        url, status_code = response.redirect_chain[-1]
        self.assertEqual(response.status_code, target_status_code, msg_prefix + "Response didn't redirect as expected: Final Response code was %d (expected %d)" % (response.status_code, target_status_code))
    else:
        self.assertEqual(response.status_code, status_code, msg_prefix + "Response didn't redirect as expected: Response code was %d (expected %d)" % (response.status_code, status_code))
        url = response.url
        scheme, netloc, path, query, fragment = urlsplit(url)
        if not path.startswith('/'):
            url = urljoin(response.request['PATH_INFO'], url)
            path = urljoin(response.request['PATH_INFO'], path)
        if fetch_redirect_response:
            domain, port = split_domain_port(netloc)
            if domain and (not validate_host(domain, settings.ALLOWED_HOSTS)):
                raise ValueError("The test client is unable to fetch remote URLs (got %s). If the host is served by Django, add '%s' to ALLOWED_HOSTS. Otherwise, use assertRedirects(..., fetch_redirect_response=False)." % (url, domain))
            extra = response.client.extra or {}
            headers = response.client.headers or {}
            redirect_response = response.client.get(path, QueryDict(query), secure=scheme == 'https', headers=headers, **extra)
            self.assertEqual(redirect_response.status_code, target_status_code, msg_prefix + "Couldn't retrieve redirection page '%s': response code was %d (expected %d)" % (path, redirect_response.status_code, target_status_code))
    self.assertURLEqual(url, expected_url, msg_prefix + "Response redirected to '%s', expected '%s'" % (url, expected_url))