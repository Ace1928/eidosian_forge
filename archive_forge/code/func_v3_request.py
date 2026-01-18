import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def v3_request(self, path, **kwargs):
    if kwargs.pop('noauth', None):
        return self.v3_noauth_request(path, **kwargs)
    auth_arg = kwargs.pop('auth', None)
    if auth_arg:
        token = self.get_requested_token(auth_arg)
    else:
        token = kwargs.pop('token', None)
        if not token:
            token = self.get_scoped_token()
    path = '/v3' + path
    return self.admin_request(path=path, token=token, **kwargs)