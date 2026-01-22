import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
def _application(self):
    """A base wsgi application that returns a simple response."""

    def app(environ, start_response):
        body = uuid.uuid4().hex.encode('utf-8')
        resp_headers = [('Content-Type', 'text/html; charset=utf8'), ('Content-Length', str(len(body)))]
        start_response('200 OK', resp_headers)
        return [body]
    return app